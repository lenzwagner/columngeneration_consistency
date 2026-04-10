import pandas as pd

def main():
    # --- results_comp.xlsx ---
    df_comp = pd.read_excel('results_comp.xlsx')
    df_comp['gap'] = df_comp['gap'] * 100
    results_comp = df_comp.groupby(['I', 'pattern'])['gap'].agg(['mean', 'std'])
    
    # --- results_analysis.xlsx ---
    df_cg = pd.read_excel('results_analysis.xlsx')
    df_cg['gap'] = df_cg['gap']
    
    results_cg_gap = df_cg.groupby(['I', 'pattern'])['gap'].agg(['mean', 'std'])
    results_cg_time = df_cg.groupby(['I', 'pattern'])['time_total'].agg(['mean', 'std'])
    results_cg_time_rmp = df_cg.groupby(['I', 'pattern'])['time_rmp'].agg(['mean', 'std'])
    results_cg_time_sp = df_cg.groupby(['I', 'pattern'])['time_sp'].agg(['mean', 'std'])
    results_cg_time_ip = df_cg.groupby(['I', 'pattern'])['time_ip'].agg(['mean', 'std'])
    results_cg_iter = df_cg.groupby(['I', 'pattern'])['iteration'].agg(['mean', 'std'])
    
    # LB/UB (nicht fuer die Tabelle)
    results_comp_obj = df_comp.groupby(['I', 'pattern'])[['incumbent', 'lower_bound']].agg(['mean', 'std'])
    results_cg_obj = df_cg.groupby(['I', 'pattern'])[['objval', 'lbound']].agg(['mean', 'std'])
    
    # Print LaTeX template
    print(r"""\begin{table}[ht]
\footnotesize
\centering
    \caption{Computational results for the compact model and the \ac{cg} approach.}
    \label{tab:initial_ress}
\begin{talltblr}[]{colsep = 3pt,
colspec = {@{} c *{10}{Q[c]} @{}},
  cell{1}{1} = {r=2}{},
  cell{1}{2} = {c=3}{},
  cell{1}{5} = {c=7}{}
}
\toprule 
$m$ & Compact Model & & & Column Generation & & & & & & \\
\cmidrule[lr]{2-4}
\cmidrule[lr]{5-11} 
& LB / UB & {Gap \\ (\%)\hyperlink{tab_a}{\textsuperscript{a}}} & {Time \\ (s)\hyperlink{tab_b}{\textsuperscript{b}}} & LB / UB & {Gap \\ (\%)\hyperlink{tab_c}{\textsuperscript{c}}} & {Time \\ (s)} & {Time MP \\ (s)} & {Time SP \\ (s)} & {Time IP \\ (s)} & \# Iterations \\
\midrule""")

    # Iterate over the index
    m = 1
    for (i, pattern) in results_comp.index:
        # Compact Model stats
        comp_gap_m = results_comp.loc[(i, pattern), 'mean']
        comp_gap_s = results_comp.loc[(i, pattern), 'std']
        comp_lb_m = results_comp_obj.loc[(i, pattern), ('lower_bound', 'mean')]
        comp_lb_s = results_comp_obj.loc[(i, pattern), ('lower_bound', 'std')]
        comp_ub_m = results_comp_obj.loc[(i, pattern), ('incumbent', 'mean')]
        comp_ub_s = results_comp_obj.loc[(i, pattern), ('incumbent', 'std')]
        
        # Column Generation stats
        cg_gap_m = results_cg_gap.loc[(i, pattern), 'mean']
        cg_gap_s = results_cg_gap.loc[(i, pattern), 'std']
        cg_time_m = results_cg_time.loc[(i, pattern), 'mean']
        cg_time_s = results_cg_time.loc[(i, pattern), 'std']
        
        cg_time_rmp_m = results_cg_time_rmp.loc[(i, pattern), 'mean']
        cg_time_rmp_s = results_cg_time_rmp.loc[(i, pattern), 'std']
        cg_time_sp_m = results_cg_time_sp.loc[(i, pattern), 'mean']
        cg_time_sp_s = results_cg_time_sp.loc[(i, pattern), 'std']
        cg_time_ip_m = results_cg_time_ip.loc[(i, pattern), 'mean']
        cg_time_ip_s = results_cg_time_ip.loc[(i, pattern), 'std']
        
        cg_iter_m = results_cg_iter.loc[(i, pattern), 'mean']
        cg_iter_s = results_cg_iter.loc[(i, pattern), 'std']
        cg_lb_m = results_cg_obj.loc[(i, pattern), ('lbound', 'mean')]
        cg_lb_s = results_cg_obj.loc[(i, pattern), ('lbound', 'std')]
        cg_ub_m = results_cg_obj.loc[(i, pattern), ('objval', 'mean')]
        cg_ub_s = results_cg_obj.loc[(i, pattern), ('objval', 'std')]
        
        # Formatting string
        def fmt(mean, std):
            return f"{{{mean:.1f} \\\\ ({std:.1f})}}"
            
        def fmt_gap(mean, std):
            return f"{{{mean:.2f} \\\\ ({std:.2f})}}"

        comp_gap_str = fmt_gap(comp_gap_m, comp_gap_s)
        comp_lb_ub_str = f"{{{comp_lb_m:.1f} / {comp_ub_m:.1f} \\\\ ({comp_lb_s:.1f}) / ({comp_ub_s:.1f})}}"
        
        cg_gap_str = fmt_gap(cg_gap_m, cg_gap_s)
        cg_time_str = fmt(cg_time_m, cg_time_s)
        
        cg_time_rmp_str = fmt(cg_time_rmp_m, cg_time_rmp_s)
        cg_time_sp_str = fmt(cg_time_sp_m, cg_time_sp_s)
        cg_time_ip_str = fmt(cg_time_ip_m, cg_time_ip_s)
        
        cg_iter_str = fmt(cg_iter_m, cg_iter_s)
        cg_lb_ub_str = f"{{{cg_lb_m:.1f} / {cg_ub_m:.1f} \\\\ ({cg_lb_s:.1f}) / ({cg_ub_s:.1f})}}"
        
        # Output row
        print(f"{m} & {comp_lb_ub_str} & {comp_gap_str} & \\emph{{TLR}} & {cg_lb_ub_str} & {cg_gap_str} & {cg_time_str} & {cg_time_rmp_str} & {cg_time_sp_str} & {cg_time_ip_str} & {cg_iter_str} \\\\")
        m += 1

    print(r"""\bottomrule
\end{talltblr}
\vspace{0.2cm}
\parbox{0.95\linewidth}{\scriptsize%
\flushleft{\emph{Aggregated values are reported as mean (std. dev) across all scenarios per instance.}}\\
\hypertarget{tab_a}{\textsuperscript{a}} \ac{mip}-Gap: Relative difference between the incumbent and the current best \ac{lb}. \quad
\hypertarget{tab_b}{\textsuperscript{b}} Neither scenario reached an optimal solution within the two-hour time limit, and results are therefore reported as \emph{TLR} (time limit reached). \quad \hypertarget{tab_c}{\textsuperscript{c}} Integrality Gap: Relative difference between the final \ac{rmp}-\ac{ip} solution and its \ac{rmp}-\ac{lp} relaxation. Note that this does not represent a gap to the global \ac{mip} optimum.}

\end{table}""")

if __name__ == "__main__":
    main()
