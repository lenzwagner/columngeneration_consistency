import os
import glob
import pandas as pd
import numpy as np

def generate_tikz_code():
    threshold_files = glob.glob('results/thresholds/threshold_analysis_*.csv')
    if not threshold_files:
        print("No data found.")
        return
        
    latest_file = max(threshold_files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    
    first_tau = df['tau'].iloc[0]
    df_plot = df[df['tau'] == first_tau].copy()
    df_plot['lambda'] = df_plot['lambda'].round(2)
    df_plot = df_plot.sort_values('lambda')
    
    lambdas = df_plot['lambda'].values
    nc_uc = df_plot['No-Change_Undercoverage'].values
    unr_uc = df_plot['Unrestricted_Undercoverage'].values
    unr_sc = df_plot['Unrestricted_SC'].values
    
    red_pct = np.zeros_like(nc_uc)
    mask = nc_uc > 0
    red_pct[mask] = (nc_uc[mask] - unr_uc[mask]) / nc_uc[mask] * 100
    
    def to_coords(x_arr, y_arr):
        return " ".join([f"({x:.2f},{y:.2f})" for x, y in zip(x_arr, y_arr)])
    
    tex_code = f"""
\\begin{{figure*}}[t]
    \\centering
    
    % Panel A: Absolute Service Shortfall
    \\begin{{subfigure}}[b]{{0.32\\textwidth}}
        \\centering
        \\begin{{tikzpicture}}
            \\begin{{axis}}[
                width=\\linewidth,
                height=5cm,
                xlabel={{Demand Multiplier ($\\lambda$)}},
                ylabel={{Mean Undercoverage}},
                grid=major,
                legend pos=north west,
                legend style={{font=\\scriptsize}},
                tick label style={{font=\\scriptsize}},
                label style={{font=\\footnotesize}}
            ]
            \\addplot[color=red, dashed, mark=*, mark size=1.5pt] coordinates {{
                {to_coords(lambdas, nc_uc)}
            }};
            \\addlegendentry{{No-Change}}
            
            \\addplot[color=blue, mark=square*, mark size=1.5pt] coordinates {{
                {to_coords(lambdas, unr_uc)}
            }};
            \\addlegendentry{{Unrestricted BAP}}
            \\end{{axis}}
        \\end{{tikzpicture}}
        \\caption{{Absolute Service Shortfall}}
        \\label{{fig:thresh_A}}
    \\end{{subfigure}}
    \\hfill
    % Panel B: Percentage Reduction
    \\begin{{subfigure}}[b]{{0.32\\textwidth}}
        \\centering
        \\begin{{tikzpicture}}
            \\begin{{axis}}[
                width=\\linewidth,
                height=5cm,
                xlabel={{Demand Multiplier ($\\lambda$)}},
                ylabel={{Reduction in Undercoverage (\\%)}},
                grid=major,
                ymin=0, ymax=110,
                tick label style={{font=\\scriptsize}},
                label style={{font=\\footnotesize}}
            ]
            \\addplot[color=orange, thick, mark=diamond*, mark size=2pt] coordinates {{
                {to_coords(lambdas, red_pct)}
            }};
            
            % Annotation Node
            \\node[anchor=south west, align=center, font=\\scriptsize] at (axis cs:0.4, 99.3) {{Maximal Value}};
            \\end{{axis}}
        \\end{{tikzpicture}}
        \\caption{{Diminishing Returns}}
        \\label{{fig:thresh_B}}
    \\end{{subfigure}}
    \\hfill
    % Panel C: Shift Changes
    \\begin{{subfigure}}[b]{{0.32\\textwidth}}
        \\centering
        \\begin{{tikzpicture}}
            \\begin{{axis}}[
                width=\\linewidth,
                height=5cm,
                xlabel={{Demand Multiplier ($\\lambda$)}},
                ylabel={{Total Shift Changes}},
                grid=major,
                tick label style={{font=\\scriptsize}},
                label style={{font=\\footnotesize}}
            ]
            \\addplot[color=darkgray, thick, mark=triangle*, mark size=2.5pt] coordinates {{
                {to_coords(lambdas, unr_sc)}
            }};
            \\end{{axis}}
        \\end{{tikzpicture}}
        \\caption{{Zero-Sum Reallocations}}
        \\label{{fig:thresh_C}}
    \\end{{subfigure}}
    
    \\caption{{Behavioral scheduling dynamics under increasing demand pressure. Panel (a) shows how pure consistency (No-Change) leads to massive undercoverage. Panel (b) highlights the peak value of behavioral flexibility before structural capacity limits dominate. Panel (c) illustrates the model's rational avoidance of disruptive shift changes once the workforce is fully utilized.}}
    \\label{{fig:threshold_dynamics}}
\\end{{figure*}}
"""
    
    with open('results/plots/threshold_plots.tex', 'w') as f:
        f.write(tex_code)
    print("LaTeX TikZ code successfully written to results/plots/threshold_plots.tex")

if __name__ == '__main__':
    generate_tikz_code()
