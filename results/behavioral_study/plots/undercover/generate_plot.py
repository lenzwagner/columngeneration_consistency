import pandas as pd
import os
import ast
import numpy as np

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
    result = np.convolve(arr, np.ones(w) / w, mode='same')
    result[0] = arr[:2].mean()
    result[-1] = arr[-2:].mean()
    return result

def generate_undercoverage_plot(df):
    days = list(range(1, T + 1))
    shifts = [1, 2, 3]

    def aggregate_daily(col_name):
        seed_rows = []
        for raw in df[col_name].dropna():
            try:
                d = ast.literal_eval(raw) if isinstance(raw, str) else raw
                row_vals = [sum(d.get((day, s), 0.0) for s in shifts) for day in days]
                seed_rows.append(row_vals)
            except Exception:
                continue
        return np.array(seed_rows)

    arr_bap = aggregate_daily('shift_undercover_behavior')
    arr_npp = aggregate_daily('shift_undercover_naive')

    if arr_bap.size == 0 or arr_npp.size == 0:
        print("Warning: Could not aggregate shift_undercover data.")
        return

    mean_bap = arr_bap.mean(axis=0)
    std_bap  = arr_bap.std(axis=0)
    mean_npp = arr_npp.mean(axis=0)
    std_npp  = arr_npp.std(axis=0)

    trend_bap = moving_avg(mean_bap, w=3)
    trend_npp = moving_avg(mean_npp, w=3)

    bap_coords = fmt_err(mean_bap, std_bap)
    npp_coords = fmt_err(mean_npp, std_npp)
    trend_bap_coords = fmt_coords(trend_bap, precision=2)
    trend_npp_coords = fmt_coords(trend_npp, precision=2)

    raw_ymax = max((mean_bap + std_bap).max(), (mean_npp + std_npp).max()) - 5
    ymax = int(np.ceil(raw_ymax / 25.0) * 25) + 5

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
		ylabel={\footnotesize Absolute Undercoverage},
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
\caption{Daily absolute undercoverage for the base setting across $\mathcal{S}_5$. Error bars: standard deviation. Green/red shading: \ac{bap} advantage/investment phase.}
\label{fig:relunder}
\end{figure}
"""
    output_path = os.path.join(os.path.dirname(__file__), 'undercoverage_plot.tex')
    with open(output_path, 'w') as f:
        f.write(tikz_code)
    print(f"Undercoverage TikZ plot written to {output_path}")

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
    generate_undercoverage_plot(df)

if __name__ == "__main__":
    main()
