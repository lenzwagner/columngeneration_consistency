import pandas as pd
from Utils.gcutil import create_individual_working_list
import openpyxl

# Sets
work = pd.read_excel(r'./data/Arzt.xlsx', sheet_name='Arzt')
df = pd.read_excel(r'./data/NF.xlsx', sheet_name='NF')
df1 = pd.read_excel(r'./data/NF.xlsx', sheet_name='Shift')
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