import pandas as pd
import numpy as np
from Utils.gcutil import create_individual_working_list

# --- Hardcoded Data (formerly from Arzt.xlsx and NF.xlsx) ---
ARZT_RECORDS = [
    {'Id': 1, 'WT': 40.0, 'Weekend': 1.0},
    {'Id': 2, 'WT': 40.0, 'Weekend': 1.0},
    {'Id': 3, 'WT': 40.0, 'Weekend': 1.0},
    {'Id': 4, 'WT': 40.0, 'Weekend': 1.0},
    {'Id': 5, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 6, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 7, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 8, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 9, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 10, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 11, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 12, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 13, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 14, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 15, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 16, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 17, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 18, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 19, 'WT': np.nan, 'Weekend': np.nan},
    {'Id': 20, 'WT': np.nan, 'Weekend': np.nan}
]

NF_RECORDS = [
    {'Day': 1, 1: 2, 2: 1, 3: 0},
    {'Day': 2, 1: 1, 2: 2, 3: 0},
    {'Day': 3, 1: 1, 2: 1, 3: 1},
    {'Day': 4, 1: 1, 2: 2, 3: 0},
    {'Day': 5, 1: 2, 2: 0, 3: 1},
    {'Day': 6, 1: 1, 2: 1, 3: 1},
    {'Day': 7, 1: 0, 2: 3, 3: 0},
    {'Day': 8, 1: 2, 2: 1, 3: 0},
    {'Day': 9, 1: 0, 2: 3, 3: 0},
    {'Day': 10, 1: 1, 2: 1, 3: 1},
    {'Day': 11, 1: 3, 2: 0, 3: 0},
    {'Day': 12, 1: 0, 2: 2, 3: 1},
    {'Day': 13, 1: 1, 2: 1, 3: 1},
    {'Day': 14, 1: 2, 2: 1, 3: 0}
]

SHIFT_RECORDS = [
    {'Shift': 1, 'Hours': 8},
    {'Shift': 2, 'Hours': 8},
    {'Shift': 3, 'Hours': 8}
]

# Sets
work = pd.DataFrame(ARZT_RECORDS)
df = pd.DataFrame(NF_RECORDS)
df1 = pd.DataFrame(SHIFT_RECORDS)
I = work['Id'].tolist()
W_I = work['Weekend'].tolist()
T = df['Day'].tolist()
K = df1['Shift'].tolist()
S_T = df1['Hours'].tolist()
I_T = work['WT'].tolist()

# Zip sets
S_T = {a: c for a, c in zip(K, S_T)}
I_T = {a: d for a, d in zip(I, I_T)}
W_I = {a: e for a, e in zip(I, W_I)}

# Individual working days
Max_WD_i = create_individual_working_list(len(I), 5, 6, 5)
Min_WD_i = create_individual_working_list(len(I), 3, 4, 3)
Min_WD_i = {a: f for a, f in zip(I, Min_WD_i)}
Max_WD_i = {a: g for a, g in zip(I, Max_WD_i)}