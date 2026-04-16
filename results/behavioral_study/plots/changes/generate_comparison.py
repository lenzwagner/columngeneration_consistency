import pandas as pd
import numpy as np
import ast
import os

def generate_comparison_plot():
    # 1. Load data
    possible_paths = [
        'results_cg.xlsx',
        '../../results_cg.xlsx',
        '../results_cg.xlsx'
    ]
    file_path = None
    for p in possible_paths:
        abs_p = os.path.join(os.path.dirname(__file__), p)
        if os.path.exists(abs_p):
            file_path = abs_p
            break
            
    if not file_path:
        print("Error: results_cg.xlsx not found.")
        return
        
    df = pd.read_excel(file_path)
    if df.empty:
        print("Error: Excel file is empty.")
        return

    # --- PART 1: Find Representative (Median) Scenario ---
    def get_row_mean(row, col_name):
        try:
            val = row[col_name]
            lst = ast.literal_eval(val) if isinstance(val, str) else val
            if lst and len(lst) > 0:
                return np.mean(lst)
        except:
            pass
        return None

    df['bap_mean'] = df.apply(lambda r: get_row_mean(r, 'p_list_behavior'), axis=1)
    
    # Filter for 100 workers as requested
    df_100 = df[df['I'] == 100]
    if df_100.empty:
        print("Warning: No scenarios with 100 workers found. Using all scenarios.")
        valid_df = df.dropna(subset=['bap_mean'])
    else:
        valid_df = df_100.dropna(subset=['bap_mean'])
    
    if valid_df.empty:
        print("Warning: No valid performance data found for median selection.")
        # Fallback to first row
        row = df.iloc[0]
    else:
        overall_median = valid_df['bap_mean'].median()
        row = valid_df.iloc[(valid_df['bap_mean'] - overall_median).abs().argsort()[:1]].iloc[0]
        
    pattern = row.get('pattern', row.get('Pattern', 'N/A'))
    scenario = row.get('scenario', row.get('Scenario', 'N/A'))
    I = int(row['I'])
    T = int(row['T'])
    
    print(f"Selected Representative Scenario: Pattern={pattern}, Scenario={scenario} (Mean BAP={row['bap_mean']:.4f})")
    print(f"Dimensions: {I} Workers, {T} Days")

    # --- PART 2: Generate Comparison Grid ---
    def parse_list(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return []
        return val if isinstance(val, list) else []

    cons_b = parse_list(row['cons_list_behavior'])
    cons_n = parse_list(row['cons_list_naive'])
    
    if len(cons_b) < I * T or len(cons_n) < I * T:
        print(f"Warning: List length mismatch. Expected {I*T}, got BAP={len(cons_b)}, NPP={len(cons_n)}")
        if not cons_b or not cons_n: return

    grid_b = np.array(cons_b[:I*T]).reshape(I, T)
    grid_n = np.array(cons_n[:I*T]).reshape(I, T)
    
    # Grid: Shape (T, I)
    comp_grid = np.zeros((T, I), dtype=int)
    for worker in range(I):
        for day in range(T):
            b = 1 if grid_b[worker, day] > 0.5 else 0
            n = 1 if grid_n[worker, day] > 0.5 else 0
            
            if b == 1 and n == 1: val = 3   # Both
            elif b == 1: val = 1            # Only BAP
            elif n == 1: val = 2            # Only NPP
            else: val = 0                   # None
            
            comp_grid[day, worker] = val

    # --- PART 3: Save TXT Output ---
    output_dir = os.path.dirname(__file__)
    txt_path = os.path.join(output_dir, 'comparison_grid.txt')
    with open(txt_path, 'w') as f:
        for d in range(T):
            f.write("".join(map(str, comp_grid[d, :])) + "\n")
    print(f"Grid saved to {txt_path}")

    # --- PART 4: Generate LaTeX/TikZ Plot ---
    tikz_path = os.path.join(output_dir, 'comparison_plot.tex')
    
    # Color definitions and TikZ template
    color_defs = r"""
\definecolor{color0}{HTML}{FFFFFF} % White (None)
\definecolor{color1}{HTML}{F28C28} % customOrange2 (BAP only)
\definecolor{color2}{HTML}{007FFF} % customBlue2 (NPP only)
\definecolor{color3}{HTML}{000000} % Black (Both)
"""
    
    # Scale width based on workers to keep it proportional
    # Base T=28, I=50 -> 0.2cm cells
    cell_size = 0.15 # cm
    if I > 100: cell_size = 0.08
    elif I > 50: cell_size = 0.12

    tikz_code = r"""
\begin{figure}[htbp]
	\centering
	\begin{tikzpicture}[x=""" + str(cell_size) + r"""cm, y=-""" + str(cell_size) + r"""cm]
		% Background grid / Frame
		\draw[lightgray, ultra thin] (0,0) grid (""" + str(I) + "," + str(T) + r""");
		\draw[thick] (0,0) rectangle (""" + str(I) + "," + str(T) + r""");
"""
    
    # Collect coordinates for each color to use fewer \fill commands
    coords_by_val = {1: [], 2: [], 3: []}
    for d in range(T):
        for i in range(I):
            val = comp_grid[d, i]
            if val in coords_by_val:
                coords_by_val[val].append((i, d))

    # Add fill commands
    for val, color in [(1, 'color1'), (2, 'color2'), (3, 'color3')]:
        if coords_by_val[val]:
            tikz_code += f"\n\t\t\\fill[{color}] "
            fill_list = [f"({x},{y}) rectangle +(1,1)" for x, y in coords_by_val[val]]
            tikz_code += "\n\t\t\t" + "\n\t\t\t".join(fill_list) + ";"

    tikz_code += r"""

		% Labels
		\node[anchor=east, font=\scriptsize] at (-0.5, """ + str(T/2) + r""") {Days ($1 \dots """ + str(T) + r"""$)};
		\node[anchor=north, font=\scriptsize] at (""" + str(I/2) + r""", """ + str(T + 0.5) + r""") {Workers ($1 \dots """ + str(I) + r"""$)};
		
		% Legend
		\begin{scope}[shift={(0, """ + str(T + 2.5) + r""")}, x=1cm, y=1cm]
			\fill[color1] (0,0) rectangle (0.3, 0.3) node[anchor=west, font=\tiny, black] at (0.35, 0.15) {\acl{bap} only};
			\fill[color2] (2,0) rectangle (0.3, 0.3) node[anchor=west, font=\tiny, black] at (2.35, 0.15) {\acl{npp} only};
			\fill[color3] (4,0) rectangle (0.3, 0.3) node[anchor=west, font=\tiny, black] at (4.35, 0.15) {Both Approaches};
		\end{scope}
	\end{tikzpicture}
	\caption{Consistency comparison map for a representative scenario (Pattern=""" + str(pattern) + """, Scenario=""" + str(scenario) + r"""). Each cell represents a day/worker combination where a shift change occurred.}
	\label{fig:consistency_map}
\end{figure}
"""

    with open(tikz_path, 'w') as f:
        f.write("% Color Definitions (include in preamble if needed)\n")
        f.write(color_defs)
        f.write(tikz_code)
        
    print(f"LaTeX Plot saved to {tikz_path}")

if __name__ == "__main__":
    generate_comparison_plot()
