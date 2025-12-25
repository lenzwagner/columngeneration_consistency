import pandas as pd
import os

def generate_latex():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Updated path to match user's folder structure
    file_path = os.path.join(script_dir, 'Compact', 'Results_Analysis_Compact.xlsx')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Read the analyzed results
    # Index cols are I, pattern, Metric. 
    # Columns are stats: mean, median, min, max, etc.
    df = pd.read_excel(file_path, index_col=[0, 1, 2])
    
    # We want to extract specific metrics for the table
    # For the paper, usually: I | Pattern | Gap (Median [Min-Max]) | Time (Median [Min-Max]) | Incumbent (Median)
    
    # Get unique instances (I, Pattern)
    instances = df.index.droplevel(2).unique()
    
    latex_rows = []
    
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{ll ccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{$|\\mathcal{I}|$} & \\multirow{2}{*}{Pattern} & \\multicolumn{3}{c}{Compact Model} \\\\")
    print("\\cmidrule(lr){3-5}")
    print(" & & Gap [\\%] & Time [s] & Obj. Val \\\\")
    print("\\midrule")
    
    for i_val, pattern in instances:
        # Extract metrics for this instance
        try:
            gap_stats = df.loc[(i_val, pattern, 'gap')]
            time_stats = df.loc[(i_val, pattern, 'total_time')]
            obj_stats = df.loc[(i_val, pattern, 'incumbent')] # or 'lower_bound' if preferred
            
            # Format Gap: Median (Min-Max)
            # Gap is usually ratio, so *100 for percent
            gap_median = gap_stats['median'] * 100
            gap_min = gap_stats['min'] * 100
            gap_max = gap_stats['max'] * 100
            
            gap_str = f"{gap_median:.1f} ({gap_min:.1f}-{gap_max:.1f})"
            
            # Format Time
            time_median = time_stats['median']
            # If time is 7200 everywhere, might just say "TL"
            if time_median >= 7200:
                time_str = "TL"
            else:
                time_str = f"{time_median:.1f}"
                
            # Format Obj
            obj_median = obj_stats['median']
            obj_str = f"{obj_median:.1f}"
            
            row = f"{i_val} & {pattern} & {gap_str} & {time_str} & {obj_str} \\\\"
            print(row)
            
        except KeyError as e:
            print(f"% Missing data for {i_val}-{pattern}: {e}")
            
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\caption{Computational results for Compact Model across 25 seeds. Medians and (Min-Max) ranges reported.}")
    print("\\label{tab:compact_stochastic}")
    print("\\end{table}")

if __name__ == "__main__":
    generate_latex()
