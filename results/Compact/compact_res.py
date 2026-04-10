import pandas as pd
import numpy as np
import os

def analyze_results():
    # Use absolute path or relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'results_comp_agg.xlsx')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Reading {file_path}...")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Clean up column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    # Define metrics as requested
    metrics_to_analyze = ["lower_bound", "incumbent", "gap", "status", "undercoverage", "total_time"]
    group_cols = ["I", "pattern"]

    # Check if columns exist
    missing_cols = [c for c in metrics_to_analyze + group_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: The following expected columns were not found in the Excel file: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        # Filter out missing metrics to proceed with what's available
        metrics_to_analyze = [m for m in metrics_to_analyze if m in df.columns]
        
        # If group columns are missing, we can't really group as requested
        if any(g not in df.columns for g in group_cols):
             print("Critical grouping columns missing. Aborting.")
             return

    # Define aggregation functions
    # mean, median, min, max, spannweite (range), stnadardabweichung (std), varianz (var), IQR 25, IQR 75
    
    def range_func(x):
        return x.max() - x.min()

    def q25(x):
        return x.quantile(0.25)
        
    def q75(x):
        return x.quantile(0.75)

    agg_funcs = [
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('spannweite', range_func),
        ('std', 'std'),
        ('var', 'var'),
        ('25%', q25),
        ('75%', q75)
    ]

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.4f}'.format)

    all_results = []
    
    for metric in metrics_to_analyze:
        print(f"Processing METRIC: {metric}")
        
        # Check if data is numeric
        if not pd.api.types.is_numeric_dtype(df[metric]):
            print(f"Column '{metric}' is not numeric. Attempting to convert...")
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
        
        try:
            # Group by and aggregate
            result = df.groupby(group_cols)[metric].agg(agg_funcs)
            # Add metric name as a column to perform pivot/stacking later
            result['Metric'] = metric
            all_results.append(result)
        except Exception as e:
            print(f"Error analyzing {metric}: {e}")

    if all_results:
        print("Combining results...")
        final_df = pd.concat(all_results)
        
        # Reset index to make 'I' and 'pattern' regular columns, and ensure 'Metric' is available
        final_df = final_df.reset_index()
        
        # Set new index to include Metric, so we have I, Pattern, Metric as rows
        final_df.set_index(['I', 'pattern', 'Metric'], inplace=True)
        
        output_file = os.path.join(script_dir, 'results_comp_agg.xlsx')
        try:
            final_df.to_excel(output_file)
            print(f"\nSuccessfully wrote analysis results to: {output_file}")
            print("Preview of the data:")
            print(final_df.head(10))
        except Exception as e:
            print(f"Error writing to Excel: {e}")
    else:
        print("No results to write.")

if __name__ == "__main__":
    analyze_results()
