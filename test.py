import pandas as pd

df = pd.read_excel("results/Results.xlsx", engine="openpyxl")
gaps = df.loc[0, "perf_list_naive"]
print(len(gaps), gaps)

