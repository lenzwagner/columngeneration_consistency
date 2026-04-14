import pandas as pd
import os
import ast
import numpy as np
import subprocess
import sys

# -- Shared constants --
T = 28 # planning-horizon length in days

def main():
    # Load the file
    file_path = 'results_analysis.xlsx'
    
    # If the script is run directly in the behavioral_study folder, the Excel might be in the computational_study folder
    # Or in the root directory of the repo
    possible_paths = [
        'results_analysis.xlsx',
        'results/behavioral_study/results_analysis.xlsx',
        '../computational_study/results_analysis.xlsx'
    ]
    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if not file_path:
        print(f"Error: File not found. Checked: {possible_paths}")
        return
            
    df = pd.read_excel(file_path)
    
    metrics = [
        'undercover',
        'undercover_norm',
        'cons',
        'cons_norm',
        'perf',
        'perf_norm',
        'understaffing',
        'understaffing_norm',
        'spread_sc',
        'load_share_sc',
        'gini_sc',
        'top10_sc'
    ]
    
    print("=== Comparative Analysis: behavioral_study vs Naive ===\n")
    
    summary_dfs = []
    
    for metric in metrics:
        col_b = f"{metric}_behavior"
        col_n = f"{metric}_naive"
        
        if col_b in df.columns and col_n in df.columns:
            stats = df[[col_b, col_n]].agg(['min', 'max', 'median', 'mean', 'std']).T
            
            if metric in ['load_share_sc', 'top10_sc']:
                stats = stats * 100
                
            summary_dfs.append(stats)
            
            print(f"--- Metrik: {metric} ---")
            print(stats.to_string())
            print("\n")
        else:
            print(f"Warning: Columns for '{metric}' not found in the file.\n")
            
    if summary_dfs:
        final_df = pd.concat(summary_dfs)
        final_df = final_df.round(2)
        
        final_df.reset_index(inplace=True)
        final_df.rename(columns={'index': 'metric'}, inplace=True)
        
        # Addition: Calculate and insert reduction
        try:
            mean_naive = final_df.loc[final_df['metric'] == 'undercover_naive', 'mean'].values[0]
            mean_behavior = final_df.loc[final_df['metric'] == 'undercover_behavior', 'mean'].values[0]
            rel_reduction = (mean_naive - mean_behavior) / mean_naive
            
            new_row = pd.DataFrame([{'metric': 'reduction', 'min': round(rel_reduction * 100, 2)}])
            final_df = pd.concat([final_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Could not append 'reduction': {e}")
        
        output_file = 'results_analysis_summary.csv'
        final_df.to_csv(output_file, index=False)
        print(f"Statistics were exported to: {output_file}\n")
    
    # === Sub-Plot Generations ===
    scripts = [
        os.path.join('plots', 'performance', 'generate_plot.py'),
        os.path.join('plots', 'undercover', 'generate_plot.py'),
        os.path.join('plots', 'changes', 'generate_plot.py')
    ]
    
    for script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(script_path):
            print(f"Running sub-script: {script}...")
            subprocess.run([sys.executable, script_path], check=True)
        else:
            print(f"Warning: Script not found: {script_path}")

if __name__ == "__main__":
    main()
