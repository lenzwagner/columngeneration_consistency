import pandas as pd
import os
import ast
import numpy as np

# ── Shared constants ────────────────────────────────────────────────────────
T = 28  # planning-horizon length in days

# ── Shared helpers ───────────────────────────────────────────────────────────
def fmt_coords(arr, precision=3):
    """Format a 1-D array as pgfplots coordinate pairs starting at day 1."""
    fmt = f"{{:.{precision}f}}"
    return " ".join([f"({i + 1}, {fmt.format(v)})" for i, v in enumerate(arr)])

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
    
    # === TikZ Plot Export ===
    generate_tikz_plot(df)

    # === Undercoverage Plot Export ===
    generate_undercoverage_plot(df)


def generate_tikz_plot(df):
    def get_stats(col_name):
        # Collect all lists from the column
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

        # Each list consists of T days for 100 workers
        # Structure: [W1_D1, W1_D2, ..., W1_D28, W2_D1, ...]
        # We want the statistics per day (1-T) across ALL workers in ALL scenarios
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

    # Auxiliary values for arrows (Day 14 and 26)
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

    with open('performance_plot.tex', 'w') as f:
        f.write(tikz_code)
    print("TikZ Plot was exported to 'performance_plot.tex'.")


def generate_undercoverage_plot(df):
    """Generate a TikZ plot of daily relative undercoverage (sum of shifts per day),
    averaged and std across all seeds. BAP = peachy/orange, NPP = purply.
    Writes 'undercoverage_plot.tex' next to this script.
    """
    days = list(range(1, T + 1))
    shifts = [1, 2, 3]

    def aggregate_daily(col_name):
        """Return (n_seeds x T) numpy array of daily summed undercoverage."""
        seed_rows = []
        for raw in df[col_name].dropna():
            try:
                d = ast.literal_eval(raw) if isinstance(raw, str) else raw
                row_vals = [sum(d.get((day, s), 0.0) for s in shifts) for day in days]
                seed_rows.append(row_vals)
            except Exception:
                continue
        return np.array(seed_rows)  # shape: (n_seeds, T)

    arr_bap = aggregate_daily('shift_undercover_behavior')
    arr_npp = aggregate_daily('shift_undercover_naive')

    if arr_bap.size == 0 or arr_npp.size == 0:
        print("Warning: Could not aggregate shift_undercover data for undercoverage plot.")
        return

    mean_bap = arr_bap.mean(axis=0)
    std_bap  = arr_bap.std(axis=0)
    mean_npp = arr_npp.mean(axis=0)
    std_npp  = arr_npp.std(axis=0)

    # ------------------------------------------------------------------ helpers
    def fmt_err(means, stds):
        """Format coordinates with +- error bars for pgfplots."""
        parts = [f"({i + 1}, {m:.2f}) +- (0, {s:.2f})" for i, (m, s) in enumerate(zip(means, stds))]
        return "\n\t\t\t\t" + "\n\t\t\t\t".join(parts)

    # Simple moving-average trend (window=3, reflect at edges)
    def moving_avg(arr, w=3):
        result = np.convolve(arr, np.ones(w) / w, mode='same')
        result[0] = arr[:2].mean()
        result[-1] = arr[-2:].mean()
        return result

    trend_bap = moving_avg(mean_bap, w=3)
    trend_npp = moving_avg(mean_npp, w=3)

    bap_coords = fmt_err(mean_bap, std_bap)
    npp_coords = fmt_err(mean_npp, std_npp)
    trend_bap_coords = fmt_coords(trend_bap, precision=2)
    trend_npp_coords = fmt_coords(trend_npp, precision=2)

    # Determine ymax (round up to next multiple of 25 above max+std)
    raw_ymax = max((mean_bap + std_bap).max(), (mean_npp + std_npp).max())
    ymax = int(np.ceil(raw_ymax / 25.0) * 25) + 5  # a bit of padding

    # Build ytick string dynamically
    ytick_vals = [0, 25, 50, 75]
    while ytick_vals[-1] + 25 <= ymax:
        ytick_vals.append(ytick_vals[-1] + 25)
    ytick_str = ",".join(str(v) for v in ytick_vals)

    tikz_code = r"""\begin{figure}
\begin{tikzpicture}
\begin{axis}[
		width=\textwidth,
		height=0.35\textwidth,
		xlabel={\footnotesize Day},
		ylabel={\footnotesize Relative Undercoverage},
		ymin=0,
		ymax=""" + str(ymax) + r""",
		xmin=0.5,
		xmax=28.5,
		xtick={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28},
		ytick={""" + ytick_str + r"""},
		ymajorgrids=true,
		grid style=dashed,
		enlarge x limits=0.03,
		tick label style={font=\scriptsize},
		xtick pos=bottom,
		ytick pos=left,
		legend style={
			at={(0.02,0.98)},
			anchor=north west,
			legend columns=-1,
			draw=none,
			fill=white,
			font=\footnotesize
		},
	]

	\definecolor{peachy}{HTML}{febb98}
	\definecolor{purply}{HTML}{75569a}
	\definecolor{lightgreen}{HTML}{90EE90}
	\definecolor{lightred}{HTML}{FFB6B6}

	% --- BAP data ---
	\addplot[
		name path=BAP,
		color=peachy,
		line width=1.5pt,
		mark=*,
		mark size=1.2pt,
		error bars/.cd,
		y dir=both,
		y explicit,
		error bar style={peachy, line width=0.5pt}
	] coordinates {""" + bap_coords + r"""
	};

	% --- NPP data ---
	\addplot[
		name path=NPP,
		color=purply,
		line width=1.5pt,
		mark=square*,
		mark size=1.2pt,
		error bars/.cd,
		y dir=both,
		y explicit,
		error bar style={purply, line width=0.5pt}
	] coordinates {""" + npp_coords + r"""
	};

	% --- Shading: green where BAP is better (NPP > BAP), days 7-28 ---
	\addplot[lightgreen!60, opacity=0.4] fill between[
		of=BAP and NPP,
		soft clip={domain=7:28}
	];

	% --- Shading: red where BAP is worse (BAP > NPP), days 1-6 investment phase ---
	\addplot[lightred!60, opacity=0.4] fill between[
		of=BAP and NPP,
		soft clip={domain=1:6}
	];

	% --- Trend line BAP (smoothed) ---
	\addplot[
		peachy,
		line width=0.8pt,
		dashed,
		opacity=0.7,
		smooth
	] coordinates {""" + " " + trend_bap_coords + r"""};

	% --- Trend line NPP (smoothed) ---
	\addplot[
		purply,
		line width=0.8pt,
		dashed,
		opacity=0.7,
		smooth
	] coordinates {""" + " " + trend_npp_coords + r"""};

	\legend{\footnotesize \acl{bap}, \footnotesize \acl{npp}}
\end{axis}
\end{tikzpicture}
\vspace{-0.3cm}
\caption{Daily relative undercoverage for the base setting across $\mathcal{S}_5$. Error bars: standard deviation. Green/red shading: \ac{bap} advantage/investment phase.}
\label{fig:relunder}
\end{figure}
"""

    out_file = 'undercoverage_plot.tex'
    with open(out_file, 'w') as f:
        f.write(tikz_code)
    print(f"Undercoverage TikZ plot written to '{out_file}'.")
    print(f"  Seeds used  — BAP: {len(arr_bap)}, NPP: {len(arr_npp)}")
    print(f"  y-axis range: 0 – {ymax}")
    print(f"  Day means (BAP / NPP):")
    for d in range(T):
        print(f"    Day {d+1:2d}: BAP={mean_bap[d]:.2f} ± {std_bap[d]:.2f}  "
              f"| NPP={mean_npp[d]:.2f} ± {std_npp[d]:.2f}")


if __name__ == "__main__":
    main()
