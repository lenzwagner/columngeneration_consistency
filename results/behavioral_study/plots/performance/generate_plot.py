import pandas as pd
import os
import ast
import numpy as np

# --- Constants ---
T = 28  # planning-horizon length in days

# --- Helpers ---
def fmt_coords(arr, precision=3):
    """Format a 1-D array as pgfplots coordinate pairs starting at day 1."""
    fmt = f"{{:.{precision}f}}"
    return " ".join([f"({i + 1}, {fmt.format(v)})" for i, v in enumerate(arr)])

def generate_tikz_plot(df):
    def get_stats(col_name):
        all_lists = []
        for raw in df[col_name].dropna():
            try:
                lst = ast.literal_eval(raw) if isinstance(raw, str) else raw
                if len(lst) == 2800:
                    all_lists.append(lst)
            except:
                continue

        if not all_lists:
            return None, None, None

        n_scenarios = len(all_lists)
        all_data = np.array(all_lists).reshape(n_scenarios * 100, T)

        means = np.mean(all_data, axis=0)
        stds = np.std(all_data, axis=0)
        return means, stds, all_data

    mean_b, std_b, _ = get_stats('p_list_behavior')
    mean_n, std_n, _ = get_stats('p_list_naive')

    if mean_b is None or mean_n is None:
        print("Warning: Could not find or process p_list data for TikZ export.")
        return

    upper_b = np.clip(mean_b + std_b, 0, 1)
    lower_b = np.clip(mean_b - std_b, 0, 1)
    upper_n = np.clip(mean_n + std_n, 0, 1)
    lower_n = np.clip(mean_n - std_n, 0, 1)

    idx14, idx26 = 13, 25
    gap14 = (mean_b[idx14] - mean_n[idx14]) / mean_b[idx14] * 100
    gap26 = (mean_b[idx26] - mean_n[idx26]) / mean_b[idx26] * 100

    tikz_code = r"""\begin{figure}[ht]
	\centering
	\begin{tikzpicture}[scale=1]
		\definecolor{customOrange2}{HTML}{febb98}
		\definecolor{customBlue2}{HTML}{75569a} 
		\definecolor{safetyred}{HTML}{DC143C}
		\definecolor{recoverygreen}{HTML}{32CD32}
		
		\begin{axis}[
				width=\textwidth,
				height=0.34\textwidth,
				ymajorgrids=true,
				xlabel={\footnotesize Day},
				ylabel={\footnotesize Average Daily Performance},
				ylabel style={rotate=0},
				xmin=1.02,
				xmax=28,
				ymin=0.55,
				ymax=1.0,
				xtick={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28},
				ytick={0.6,0.7,0.8,0.9,1.0},
				major tick length=2pt,
				tick style={color=black},
				xticklabel style={yshift=0.11ex},
				yticklabel style={xshift=0.11ex},
				ticklabel style={font=\scriptsize},
				axis lines=box,
				x tick label style={
					/pgf/number format/.cd,
					precision=0,
					/tikz/.cd
				},
				legend style={
					at={(0.23,0.14)},
					anchor=center,
					legend columns=1,
					draw=black,
					fill=white,
					opacity=1,
					font=\scriptsize,
					rounded corners=3pt
				},
				tick style={black},
				x axis line style={opacity=1},
				y axis line style={opacity=1},
				axis line style={opacity=1},
				major tick style={draw=black},
				tick pos=left,
				xtick pos=bottom,
				ytick pos=left
			]
						    
			% Behavioral Scheduling Approach (orange)
			\addplot[thick, customOrange2] coordinates { """ + fmt_coords(mean_b) + r""" };
						    
			% Standard deviation BAP
			\addplot[name path=upper1, draw=none] coordinates { """ + fmt_coords(upper_b) + r""" };
			\addplot[name path=lower1, draw=none] coordinates { """ + fmt_coords(lower_b) + r""" };
			\addplot[customOrange2!30, opacity=0.6] fill between[of=upper1 and lower1];
						
			% Naive Approach (blue)
			\addplot[thick, customBlue2] coordinates { """ + fmt_coords(mean_n) + r""" };
						
			% Standard deviation Naive
			\addplot[name path=upper2, draw=none] coordinates { """ + fmt_coords(upper_n) + r""" };
			\addplot[name path=lower2, draw=none] coordinates { """ + fmt_coords(lower_n) + r""" };
			\addplot[customBlue2!30, opacity=0.6] fill between[of=upper2 and lower2];
			
			% --- GAP ARROW Tag 14 ---
			\draw[<->, thick, black!70, line width=1pt] (axis cs:14.6,""" + f"{mean_n[idx14]:.3f}" + r""") -- (axis cs:14.6,""" + f"{mean_b[idx14]:.3f}" + r""");
			\node[anchor=west, font=\scriptsize\bfseries, text=black!70] at (axis cs:14.6,""" + f"{(mean_b[idx14]+mean_n[idx14])/2:.3f}" + r""") {$\sim""" + f"{gap14:.0f}" + r"""\%$ Eff.};
			
			% --- GAP ARROW Tag 26 ---
			\draw[<->, thick, black!70, line width=1pt] (axis cs:26,""" + f"{mean_n[idx26]:.3f}" + r""") -- (axis cs:26,""" + f"{mean_b[idx26]:.3f}" + r""");
			\node[anchor=west, font=\scriptsize\bfseries, text=black!70] at (axis cs:23.04,""" + f"{(mean_b[idx26]+mean_n[idx26])/2:.3f}" + r""") {$\sim""" + f"{gap26:.0f}" + r"""\%$ Eff.};
						
			\legend{\scriptsize \acl{bap} \textcolor{white}{fffgf}, \scriptsize \acl{npp}}
		\end{axis}
	\end{tikzpicture}
	\caption{Average daily worker performance over the planning horizon. Dashed red line shows critical safety threshold (75\%). Gap arrows quantify \ac{bap} efficiency advantage. \textit{Note: ‘Eff.’ is an abbreviation for efficiency.}}
	\label{fig:perf_plot}
\end{figure}
"""
    output_path = os.path.join(os.path.dirname(__file__), 'performance_plot.tex')
    with open(output_path, 'w') as f:
        f.write(tikz_code)
    print(f"TikZ Plot exported to {output_path}")

def main():
    # If Excel is not found in the current folder, check parent folders
    possible_paths = [
        'results_analysis.xlsx',
        '../../results_analysis.xlsx',
        '../results_analysis.xlsx'
    ]
    file_path = None
    for p in possible_paths:
        abs_p = os.path.join(os.path.dirname(__file__), p)
        if os.path.exists(abs_p):
            file_path = abs_p
            break
            
    if not file_path:
        print(f"Error: results_analysis.xlsx not found.")
        return
            
    df = pd.read_excel(file_path)
    generate_tikz_plot(df)

if __name__ == "__main__":
    main()
