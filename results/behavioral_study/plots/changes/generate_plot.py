import pandas as pd
import ast
import os
import numpy as np

def generate_changes_plot():
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The excel is 2 levels up from results/behavioral_study/plots/changes/
    results_path = os.path.join(script_dir, "../../results_analysis.xlsx")
    output_path = os.path.join(script_dir, "changes_plot.tex")
    
    if not os.path.exists(results_path):
        # Fallback to local
        results_path = "results_analysis.xlsx"
        if not os.path.exists(results_path):
            print(f"Error: {results_path} not found.")
            return

    # 2. Load Data
    df = pd.read_excel(results_path)
    
    def count_differences(seq_str):
        try:
            if isinstance(seq_str, str):
                seq = ast.literal_eval(seq_str)
            else:
                seq = seq_str
            # Differences are coded as 1 (Only BAP) or 2 (Only NPP)
            return sum(1 for x in seq if x in [1, 2])
        except:
            return 0

    # 3. Find Representative Instance
    df['diff_count'] = df['changes_sequence'].apply(count_differences)
    mean_diffs = df['diff_count'].mean()
    
    # Find the row closest to the mean
    idx_closest = (df['diff_count'] - mean_diffs).abs().idxmin()
    representative_row = df.loc[idx_closest]
    
    seq_str = representative_row['changes_sequence']
    if isinstance(seq_str, str):
        full_sequence = ast.literal_eval(seq_str)
    else:
        full_sequence = seq_str
        
    # 4. Scenario Info
    num_workers_to_plot = 28
    
    # Identify total workers in this sequence
    i_val = int(representative_row.get('I', 100))
    total_slots = len(full_sequence)
    slots_per_worker = total_slots // i_val
    
    # 5. Format TikZ
    tikz_lines = []
    for i in range(min(i_val, num_workers_to_plot)):
        start = i * slots_per_worker
        end = start + slots_per_worker
        worker_seq = full_sequence[start:end]
        line_str = "".join(map(str, worker_seq))
        tikz_lines.append(f"\t\t{line_str}")

    # 6. Write Template
    template = r"""\begin{figure}[htbp]
	\centering
	\graph{%
"""
    for line in tikz_lines:
        template += line + "\n"
        
    template += r"""	}
        \vspace{-0.1cm}
\caption{Comparison of shift-change patterns between the \ac{bap} and \ac{npp} across workers and days for a representative scenario (closest to the mean total shift changes).}	\label{fig:scheduling-comparison}
\end{figure}
"""

    with open(output_path, "w") as f:
        f.write(template)
    
    print(f"Success: Generated {output_path}")
    print(f"Representative row index: {idx_closest}")
    print(f"Mean differences: {mean_diffs:.2f}, Instance differences: {representative_row['diff_count']}")

if __name__ == "__main__":
    generate_changes_plot()
