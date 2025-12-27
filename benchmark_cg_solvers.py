"""
Benchmark: CG+MIP vs CG+Labeling(fwd) vs CG+Labeling(bidir)

Uses scenarios from demand_scenarios.xlsx with:
- I=50, T=28
- eps=0.06, chi=5

Each solver runs in FRESH subprocess for fair JIT comparison.
"""

import subprocess
import sys
import time
import json
import os

# Configuration
EXCEL_PATH = 'data/demand_scenarios.xlsx'
N_SCENARIOS = 1  # Scenario 1
LEN_I = 50
LEN_T = 28
EPSILON = 0.05
CHI = 3

SOLVER_SCRIPT = '''
import sys
import time
import json
import pandas as pd
import numpy as np
from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import column_generation_behavior
import random

# Parameters from command line
scenario = int(sys.argv[1])
solver_type = sys.argv[2]
len_I = int(sys.argv[3])
len_T = int(sys.argv[4])
epsilon = float(sys.argv[5])
chi = int(sys.argv[6])
excel_path = sys.argv[7]

# Load demand from Excel
df_scenarios = pd.read_excel(excel_path, sheet_name='Sheet1')
row = df_scenarios[df_scenarios['Scenario'] == scenario].iloc[0]

# Build demand dict from Excel columns
demand_dict = {}
for d in range(1, len_T + 1):
    for s in range(1, 4):  # shifts 1,2,3
        col_name = f'{d},{s}'
        demand_dict[(d, s)] = int(row[col_name])

# Setup data
T = list(range(1, len_T + 1))
I = list(range(1, len_I + 1))
K = [1, 2, 3]

max_len = max(len(I), len(T), len(K))
data = pd.DataFrame({
    'I': I + [np.nan] * (max_len - len(I)),
    'T': T + [np.nan] * (max_len - len(T)),
    'K': K + [np.nan] * (max_len - len(K))
})

time_cg_init = 10
max_itr = 500
output_len = 98
threshold = 1e-5
time_cg = 7200

# Run CG with specified solver
t0 = time.perf_counter()
result = column_generation_behavior(
    data, demand_dict, epsilon, Min_WD_i, Max_WD_i,
    time_cg_init, max_itr, output_len, chi,
    threshold, time_cg, I, T, K, 1.0,
    sp_solver=solver_type,
    start_values=None,
    save_lp=False
)
elapsed = time.perf_counter() - t0

obj = result[8]       # IP objective
lp_bound = result[9]  # LP bound at termination
itr = result[10]
gap_pct = result[12]  # Integrality gap percentage

# Output JSON result
print(json.dumps({
    'scenario': scenario,
    'solver': solver_type,
    'obj': float(obj),
    'lp_bound': float(lp_bound),
    'gap_pct': float(gap_pct),
    'iterations': int(itr),
    'time': float(elapsed)
}))
'''

# Script for Compact MIP solver (6 min timeout)
COMPACT_SCRIPT = '''
import sys
import time
import json
import pandas as pd
import numpy as np
from Utils.compactsolver import Problem

# Parameters from command line
scenario = int(sys.argv[1])
len_I = int(sys.argv[2])
len_T = int(sys.argv[3])
epsilon = float(sys.argv[4])
chi = int(sys.argv[5])
excel_path = sys.argv[6]
time_limit = int(sys.argv[7])

# Load demand from Excel
df_scenarios = pd.read_excel(excel_path, sheet_name='Sheet1')
df_medium = df_scenarios[df_scenarios['Pattern'] == 'Medium']
row = df_medium[df_medium['Scenario'] == scenario].iloc[0]

# Build demand dict
demand_dict = {}
for d in range(1, len_T + 1):
    for s in range(1, 4):
        col_name = f'{d},{s}'
        demand_dict[(d, s)] = int(row[col_name])

# Setup data
T = list(range(1, len_T + 1))
I = list(range(1, len_I + 1))
K = [1, 2, 3]

max_len = max(len(I), len(T), len(K))
data = pd.DataFrame({
    'I': I + [np.nan] * (max_len - len(I)),
    'T': T + [np.nan] * (max_len - len(T)),
    'K': K + [np.nan] * (max_len - len(K))
})

# Solve with Compact MIP - use same constraints as CG
from Utils.setup import Min_WD_i, Max_WD_i
t0 = time.perf_counter()
problem = Problem(data, demand_dict, epsilon, Min_WD_i, Max_WD_i, chi)
problem.buildLinModel()
problem.model.Params.TimeLimit = time_limit
problem.model.Params.OutputFlag = 0
problem.model.update()
problem.model.optimize()
elapsed = time.perf_counter() - t0

incumbent = problem.model.objVal if problem.model.SolCount > 0 else float('inf')
lower_bound = problem.model.ObjBound
gap_pct = problem.model.MIPGap * 100 if problem.model.SolCount > 0 else 100.0

print(json.dumps({
    'scenario': scenario,
    'solver': 'compact',
    'obj': float(incumbent),
    'lp_bound': float(lower_bound),
    'gap_pct': float(gap_pct),
    'iterations': 0,
    'time': float(elapsed)
}))
'''

def run_single_test(scenario, solver_type, len_I, len_T, epsilon, chi, excel_path):
    """Run a single test in a fresh subprocess."""
    cmd = [
        sys.executable, '-c', SOLVER_SCRIPT,
        str(scenario), solver_type, str(len_I), str(len_T),
        str(epsilon), str(chi), excel_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=36,
            cwd='/Users/lenz/Documents/GitHub/columngeneration_consistency'
        )
        
        # Find JSON output
        for line in reversed(result.stdout.strip().split('\n')):
            if line.startswith('{'):
                return json.loads(line)
        
        return {
            'scenario': scenario,
            'solver': solver_type,
            'obj': None,
            'iterations': None,
            'time': None,
            'error': result.stderr[-500:] if result.stderr else 'No JSON output'
        }
    except subprocess.TimeoutExpired:
        return {'scenario': scenario, 'solver': solver_type, 'error': 'Timeout'}
    except Exception as e:
        return {'scenario': scenario, 'solver': solver_type, 'error': str(e)}

def run_compact_test(scenario, len_I, len_T, epsilon, chi, excel_path, time_limit=360):
    """Run Compact MIP solver in a fresh subprocess."""
    cmd = [
        sys.executable, '-c', COMPACT_SCRIPT,
        str(scenario), str(len_I), str(len_T),
        str(epsilon), str(chi), excel_path, str(time_limit)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 60,  # Extra buffer
            cwd='/Users/lenz/Documents/GitHub/columngeneration_consistency'
        )
        
        for line in reversed(result.stdout.strip().split('\n')):
            if line.startswith('{'):
                return json.loads(line)
        
        return {
            'scenario': scenario,
            'solver': 'compact',
            'obj': None,
            'lp_bound': None,
            'gap_pct': None,
            'time': None,
            'error': result.stderr[-500:] if result.stderr else 'No JSON output'
        }
    except subprocess.TimeoutExpired:
        return {'scenario': scenario, 'solver': 'compact', 'error': 'Timeout'}
    except Exception as e:
        return {'scenario': scenario, 'solver': 'compact', 'error': str(e)}

if __name__ == '__main__':
    print("=" * 80)
    print(f"Benchmark: CG+MIP vs CG+Labeling(fwd) vs CG+Labeling(bidir)")
    print(f"Config: I={LEN_I}, T={LEN_T}, eps={EPSILON}, chi={CHI}, Scenarios=1-{N_SCENARIOS}")
    print("Each test runs in fresh subprocess (fair JIT comparison)")
    print("=" * 80)
    print()
    
    # Check which scenarios exist in Excel with Medium pattern
    import pandas as pd
    df = pd.read_excel(EXCEL_PATH, sheet_name='Sheet1')
    df_medium = df[df['Pattern'] == 'Medium']
    available_scenarios = sorted(df_medium['Scenario'].unique())
    test_scenarios = [s for s in available_scenarios if s <= N_SCENARIOS]
    print(f"Available Medium scenarios: {available_scenarios[:15]}...")
    print(f"Testing scenarios: {test_scenarios}")
    print()
    
    solvers = ['labeling', 'labeling_bidir']
    results = []
    COMPACT_TIME_LIMIT = 360  # 6 minutes
    
    for scenario in test_scenarios:
        # Load and print demand for this scenario
        row = df_medium[df_medium['Scenario'] == scenario].iloc[0]
        demand_dict = {(d, s): int(row[f'{d},{s}']) for d in range(1, LEN_T + 1) for s in range(1, 4)}
        print(f"\n--- Scenario {scenario} ---")
        print(f"  Demand: {demand_dict}")
        
        # Run CG solvers
        for solver in solvers:
            print(f"  {solver}...", end=' ', flush=True)
            result = run_single_test(scenario, solver, LEN_I, LEN_T, EPSILON, CHI, EXCEL_PATH)
            results.append(result)
            
            if result.get('error'):
                print(f"ERROR: {str(result['error'])[:50]}")
            else:
                print(f"obj={result['obj']:.2f}, iter={result['iterations']}, time={result['time']:.1f}s")
        
        # Run Compact MIP solver
        print(f"  compact (6min)...", end=' ', flush=True)
        compact_result = run_compact_test(scenario, LEN_I, LEN_T, EPSILON, CHI, EXCEL_PATH, COMPACT_TIME_LIMIT)
        results.append(compact_result)
        if compact_result.get('error'):
            print(f"ERROR: {str(compact_result['error'])[:50]}")
        else:
            print(f"obj={compact_result['obj']:.2f}, LB={compact_result['lp_bound']:.2f}, gap={compact_result['gap_pct']:.1f}%")
    
    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(f"\n{'Scen':>4} | {'Fwd Time':>10} | {'Fwd Gap':>8} | {'Bid Time':>10} | {'Bid Gap':>8} | {'Compact Obj':>12} | {'Compact LB':>12} | {'Compact Gap':>11}")
    print("-" * 110)
    
    for scenario in test_scenarios:
        fwd = next((r for r in results if r['scenario'] == scenario and r['solver'] == 'labeling'), {})
        bid = next((r for r in results if r['scenario'] == scenario and r['solver'] == 'labeling_bidir'), {})
        cpt = next((r for r in results if r['scenario'] == scenario and r['solver'] == 'compact'), {})
        
        fwd_t = f"{fwd.get('time', 0):.1f}s" if fwd.get('time') else "ERR"
        fwd_g = f"{fwd.get('gap_pct', 0):.2f}%" if fwd.get('gap_pct') is not None else "ERR"
        bid_t = f"{bid.get('time', 0):.1f}s" if bid.get('time') else "ERR"
        bid_g = f"{bid.get('gap_pct', 0):.2f}%" if bid.get('gap_pct') is not None else "ERR"
        cpt_obj = f"{cpt.get('obj', 0):.1f}" if cpt.get('obj') else "ERR"
        cpt_lb = f"{cpt.get('lp_bound', 0):.1f}" if cpt.get('lp_bound') else "ERR"
        cpt_g = f"{cpt.get('gap_pct', 0):.1f}%" if cpt.get('gap_pct') is not None else "ERR"
        
        print(f"{scenario:>4} | {fwd_t:>10} | {fwd_g:>8} | {bid_t:>10} | {bid_g:>8} | {cpt_obj:>12} | {cpt_lb:>12} | {cpt_g:>11}")
    
    # Averages
    fwd_times = [r['time'] for r in results if r['solver'] == 'labeling' and r.get('time')]
    bid_times = [r['time'] for r in results if r['solver'] == 'labeling_bidir' and r.get('time')]
    
    fwd_gaps = [r['gap_pct'] for r in results if r['solver'] == 'labeling' and r.get('gap_pct') is not None]
    bid_gaps = [r['gap_pct'] for r in results if r['solver'] == 'labeling_bidir' and r.get('gap_pct') is not None]
    cpt_gaps = [r['gap_pct'] for r in results if r['solver'] == 'compact' and r.get('gap_pct') is not None]
    
    print("-" * 110)
    if fwd_times and bid_times:
        avg_fwd_t = sum(fwd_times) / len(fwd_times)
        avg_bid_t = sum(bid_times) / len(bid_times)
        
        avg_fwd_g = sum(fwd_gaps) / len(fwd_gaps) if fwd_gaps else 0
        avg_bid_g = sum(bid_gaps) / len(bid_gaps) if bid_gaps else 0
        avg_cpt_g = sum(cpt_gaps) / len(cpt_gaps) if cpt_gaps else 0
        
        print(f"{'AVG':>4} | {avg_fwd_t:>9.1f}s | {avg_fwd_g:>7.2f}% | {avg_bid_t:>9.1f}s | {avg_bid_g:>7.2f}% | {'':>12} | {'':>12} | {avg_cpt_g:>10.1f}%")
        
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"  CG+Labeling(fwd):   Avg Time = {avg_fwd_t:.2f}s, Avg Gap = {avg_fwd_g:.3f}%")
        print(f"  CG+Labeling(bidir): Avg Time = {avg_bid_t:.2f}s, Avg Gap = {avg_bid_g:.3f}%")
        print(f"  Compact MIP (6min): Avg Gap = {avg_cpt_g:.1f}%")
        print(f"  Speedup (Fwd/Bid):  {avg_fwd_t/avg_bid_t:.2f}x")