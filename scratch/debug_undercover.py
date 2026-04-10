import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import time
from datetime import datetime
from cg_behavior import column_generation_behavior
from cg_naive import column_generation_naive
from Utils.aggundercover import create_dict_from_list

# Mock data
len_I = 10
pattern = 1
scenario = 1
prob = 1.0
epsilon = 0.06
chi = 3
threshold = 0.001
time_cg = 60
time_cg_init = 60
max_itr = 2
output_len = 80
scale = 1.0

T = list(range(1, 5)) # Small T for speed
I = list(range(1, len_I + 1))
K = [1, 2, 3]

from Utils.setup import Min_WD_i, Max_WD_i # We can use these

# Mock demand_dict
demand_dict = {(t, k): 2 for t in T for k in K}

# The data used in loop_cg.py
data = pd.DataFrame({
    'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
    'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
    'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
})

# Run behavior
print("Running behavior...")
res_behavior = column_generation_behavior(
    data, demand_dict, epsilon, Min_WD_i, Max_WD_i, time_cg_init, 2, output_len, chi,
    threshold, time_cg, I, T, K, prob, sp_solver='labeling'
)

undercoverage_per_shift_behavior = res_behavior[21]
shift_undercover_behavior = create_dict_from_list(undercoverage_per_shift_behavior, len(T), len(K))
daily_undercover_behavior = {}
for (i, j), value in shift_undercover_behavior.items():
    daily_undercover_behavior[i] = daily_undercover_behavior.get(i, 0) + value

print(f"Daily behavior undercoverage: {daily_undercover_behavior}")

# Run naive
print("Running naive...")
res_naive = column_generation_naive(
    data, demand_dict, 0, Min_WD_i, Max_WD_i, time_cg_init, 2, output_len, chi,
    threshold, time_cg, I, T, K, epsilon, prob, sp_solver='labeling'
)

undercoverage_per_shift_naive = res_naive[14]
shift_undercover_naive = create_dict_from_list(undercoverage_per_shift_naive, len(T), len(K))
daily_undercover_naive = {}
for (i, j), value in shift_undercover_naive.items():
    daily_undercover_naive[i] = daily_undercover_naive.get(i, 0) + value

print(f"Daily naive undercoverage: {daily_undercover_naive}")
