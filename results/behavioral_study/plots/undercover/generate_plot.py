import pandas as pd
import os
import ast
import numpy as np

# --- Globals ---
DEMAND_CACHE = {}

# --- Constants ---
T = 28  # planning-horizon length in days

# --- Helpers ---
def fmt_coords(arr, precision=2):
    """Format a 1-D array as pgfplots coordinate pairs starting at day 1."""
    fmt = f"{{:.{precision}f}}"
    return " ".join([f"({i + 1}, {fmt.format(v)})" for i, v in enumerate(arr)])

def fmt_err(means, stds):
    """Format coordinates with +- error bars for pgfplots."""
    parts = [f"({i + 1}, {m:.2f}) +- (0, {s:.2f})" for i, (m, s) in enumerate(zip(means, stds))]
    return "\n\t\t\t\t" + "\n\t\t\t\t".join(parts)

def moving_avg(arr, w=3):
    """Simple moving-average trend (window=3, reflect at edges)."""
    if len(arr) < w: return arr
    result = np.convolve(arr, np.ones(w) / w, mode='same')
    result[0] = arr[:2].mean()
    result[-1] = arr[-2:].mean()
    return result

def load_demand_data(demand_file, pattern, scenario):
    """Load demand data for a specific pattern and scenario."""
    cache_key = (pattern, scenario)
    if cache_key in DEMAND_CACHE:
        return DEMAND_CACHE[cache_key]
    
    try:
        data = pd.read_excel(demand_file)
        # Filter rows based on Pattern and Scenario. Note: Case sensitivity may vary.
        # loop_cg uses 'Pattern' and 'Scenario' (capitalized) in the demand excel
        filtered = data[(data['Pattern'] == pattern) & (data['Scenario'] == scenario)]
        
        if filtered.empty:
            # Try lowercase if capitalized fails
            filtered = data[(data['pattern'] == pattern) & (data['scenario'] == scenario)]
            
        if not filtered.empty:
            # Extract (day, shift) -> demand from columns like '1,1', '1,2', etc.
            demand_dict = {
                tuple(map(int, col.split(','))): filtered[col].values[0]
                for col in data.columns if ',' in col
            }
            DEMAND_CACHE[cache_key] = demand_dict
            return demand_dict
    except Exception as e:
        print(f"Error loading demand data: {e}")
        
    return {}

def generate_undercoverage_plot(df, suffix='abs'):
    days = list(range(1, T + 1))
    shifts = [1, 2, 3]

    def aggregate_daily(col_name, relative=False):
        seed_rows = []
        demand_file = os.path.join(os.path.dirname(__file__), '../../../../data/demand_data.xlsx')
        
        for idx, row in df.iterrows():
            raw = row[col_name]
            if pd.isna(raw): continue
            
            try:
                d = ast.literal_eval(raw) if isinstance(raw, str) else raw
                row_vals = [sum(d.get((day, s), 0.0) for s in shifts) for day in days]
                
                if relative:
                    pattern = row.get('pattern', row.get('Pattern'))
                    scenario = row.get('scenario', row.get('Scenario'))
                    demand_dict = load_demand_data(demand_file, pattern, scenario)
                    
                    if demand_dict:
                        daily_demand = [sum(demand_dict.get((day, s), 0.0) for s in shifts) for day in days]
                        row_vals = [v / d_val if d_val > 0 else 0 for v, d_val in zip(row_vals, daily_demand)]
                    else:
                        continue # Skip if demand missing for relative plot
                
                seed_rows.append(row_vals)
            except Exception as e:
                # print(f"Error in aggregation: {e}")
                continue
        return np.array(seed_rows)

    is_rel = (suffix == 'rel')
    arr_bap = aggregate_daily('shift_undercover_behavior', relative=is_rel)
    arr_npp = aggregate_daily('shift_undercover_naive', relative=is_rel)

    if arr_bap.size == 0 or arr_npp.size == 0:
        print(f"Warning: Could not aggregate data for {suffix} plot.")
        return

    mean_bap = arr_bap.mean(axis=0)
    std_bap  = arr_bap.std(axis=0)
    mean_npp = arr_npp.mean(axis=0)
    std_npp  = arr_npp.std(axis=0)

    trend_bap = moving_avg(mean_bap, w=3)
    trend_npp = moving_avg(mean_npp, w=3)

    bap_coords = fmt_err(mean_bap, std_bap)
    npp_coords = fmt_err(mean_npp, std_npp)
    trend_bap_coords = fmt_coords(trend_bap, precision=2 if not is_rel else 4)
    trend_npp_coords = fmt_coords(trend_npp, precision=2 if not is_rel else 4)

    if is_rel:
        raw_ymax = (max((mean_bap + std_bap).max(), (mean_npp + std_npp).max())) * 1.15
        ymax = round(raw_ymax, 2)
        ytick_vals = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        ytick_vals = [v for v in ytick_vals if v <= ymax + 0.05]
        ytick_str = ",".join(str(v) for v in ytick_vals)
        ylabel = "Relative Undercoverage (\\%)"
        # For relative axis, we use percent format
        extra_axis_params = "yticklabel={\\pgfmathparse{\\tick*100}\\pgfmathprintnumber{\\pgfmathresult}\\%},"
    else:
        raw_ymax = max((mean_bap + std_bap).max(), (mean_npp + std_npp).max()) - 5
        ymax = int(np.ceil(raw_ymax / 25.0) * 25) + 5
        ytick_vals = [0, 25, 50, 75]
        while ytick_vals[-1] + 25 <= ymax:
            ytick_vals.append(ytick_vals[-1] + 25)
        ytick_str = ",".join(str(v) for v in ytick_vals)
        ylabel = "Absolute Undercoverage"
        extra_axis_params = ""

    tikz_code = r"""\begin{figure}
\begin{tikzpicture}
\begin{axis}[
		width=\textwidth,
		height=0.35\textwidth,
		xlabel={\footnotesize Day},
		ylabel={\footnotesize """ + ylabel + r"""},
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

	\node[
			anchor=north west,
			font=\footnotesize,
			fill=white,
			draw=black,
			inner sep=1.5mm,
			rounded corners
		] at (rel axis cs:0.02,0.98) {%
			\begin{tabular}{@{}c@{\hspace{1mm}}l@{\hspace{3mm}}c@{\hspace{1mm}}l@{}}
				\legendpx{1} & \acl{bap} &
				\legendpx{2} & \acl{npp}
			\end{tabular}%
		};
\end{axis}
\end{tikzpicture}
\vspace{-0.3cm}
\caption{Daily """ + ("relative" if is_rel else "absolute") + r""" undercoverage for the base setting across $\mathcal{S}_5$. Error bars: standard deviation. Green/red shading: \ac{bap} advantage/investment phase.}
\label{fig:""" + ("rel" if is_rel else "rel") + r"""under}
\end{figure}
"""
    output_path = os.path.join(os.path.dirname(__file__), f'undercoverage_{suffix}_plot.tex')
    with open(output_path, 'w') as f:
        f.write(tikz_code)
    print(f"Undercoverage TikZ plot ({suffix}) written to {output_path}")

def main():
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
    generate_undercoverage_plot(df, suffix='abs')
    generate_undercoverage_plot(df, suffix='rel')

if __name__ == "__main__":
    main()
