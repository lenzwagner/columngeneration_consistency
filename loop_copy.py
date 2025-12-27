from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import *
from cg_naive import column_generation_naive
from Utils.Plots.plots import *
from Utils.aggundercover import *
from datetime import *
from Utils.demand import *
import time

# DataFrame
results = pd.DataFrame(columns=['I', 'T', 'K', 'pattern', 'scenario', 'prob', 'epsilon', 'chi', 'gap', 'lagrange', 'objval', 'lbound', 'iteration', 'time_sp', 'time_rmp',
                                'time_ip', 'undercover_behavior', 'undercover_norm_behavior', 'cons_behavior', 'cons_norm_behavior', 'perf_behavior',
                                'perf_norm_behavior', 'understaffing_behavior', 'understaffing_norm_behavior', 'undercover_naive', 'undercover_norm_naive', 'cons_naive',
                                'cons_norm_naive', 'perf_naive', 'perf_norm_naive', 'understaffing_naive', 'understaffing_norm_naive', 'shift_undercover_naive',
                                'shift_undercover_behavior', 'perf_list_behavior', 'perf_list_naive', 'cons_list_behavior', 'cons_list_naive', 'p_list_behavior', 'p_list_naive',
                                'x_list_behavior', 'x_list_naive', 'r_list_behavior', 'r_list_naive', 'daily_undercover_behavior', 'daily_undercover_naive', 'results_ineq_sc_behavior', 'results_ineq_sc_naive', 'spread_sc_behavior',
                                'spread_sc_naive', 'load_share_sc_behavior', 'load_share_sc_naive', 'gini_sc_behavior', 'gini_sc_naive', 'results_ineq_perf_behavior',
                                'results_ineq_perf_naive', 'spread_perf_behavior', 'spread_perf_naive', 'load_share_perf_behavior', 'load_share_perf_naive', 'gini_perf_behavior',
                                'gini_perf_naive', 'shift_blocks_behavior', 'shift_blocks_naive'])

# Times and Parameter
time_Limit, time_cg, time_cg_init, prob = 7200, 7200, 10, 1.0
max_itr, output_len, mue, threshold = 200, 98, 1e-4, 6e-5

start_time = time.time()

# Store comparison results
comparison_results = []

# Loop
for epsilon in [0.06]:
    for chi in [5]:
        for len_I in [50]:
            for pattern in ['Medium']:
                for scenario in range(1, 2):  # 5 scenarios
                    if pattern == 'Medium':
                        prob = 1.0
                    elif pattern == 'High':
                        prob = 1.1
                    elif pattern == 'Low':
                        prob = 0.9

                    # Data
                    T = list(range(1, 29))
                    I = list(range(1, len_I + 1))
                    K = [1, 2, 3]

                    data = pd.DataFrame({
                        'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
                        'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
                        'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
                    })

                    random.seed = 2
                    demand_dict = generate_dict_from_excel('data/demand_scenarios.xlsx', len(I), pattern, scenario=scenario)
                    eps = epsilon

                    print(f"")
                    print(f"Iteration: Eps: {epsilon} - Chi: {chi} - I: {len(I)} - Pattern: {pattern} - K: {scenario}")
                    print(f"")

                    ## Column Generation - CG+Gurobi (MIP subproblem)
                    print('Doing behaviour with CG+Gurobi (MIP SP)')
                    t0_mip = time.time()
                    (undercoverage_behavior, understaffing_behavior, perfloss_behavior, consistency_behavior, consistency_norm_behavior, undercoverage_norm_behavior, understaffing_norm_behavior,
                     perfloss_norm_behavior, final_obj_behavior, final_lb, itr, lagrangeB, gap, time_sps, time_rmp, time_ip, ls_p_behavior, ls_sc_behavior, ls_perf_behavior, ls_x_behavior,
                     ls_r_behavior, undercoverage_per_shift_behavior, results_ineq_sc_behavior, spread_sc_behavior, load_share_sc_behavior, gini_sc_behavior, results_ineq_perf_behavior,
                     spread_perf_behavior, load_share_perf_behavior, gini_perf_behavior, shift_blocks_behavior) = column_generation_behavior(
                        data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, prob, sp_solver='mip'
                    )
                    time_mip = time.time() - t0_mip
                    print(f'  --> CG+MIP: obj={final_obj_behavior:.2f}, iter={itr}, time={time_mip:.1f}s')

                    ## Column Generation - CG+bidir (Labeling subproblem)
                    print('Doing behaviour with CG+Labeling (bidir SP)')
                    t0_bidir = time.time()
                    (undercoverage_bidir, understaffing_bidir, perfloss_bidir, consistency_bidir, consistency_norm_bidir, undercoverage_norm_bidir, understaffing_norm_bidir,
                     perfloss_norm_bidir, final_obj_bidir, final_lb_bidir, itr_bidir, lagrangeB_bidir, gap_bidir, time_sps_bidir, time_rmp_bidir, time_ip_bidir, ls_p_bidir, ls_sc_bidir, ls_perf_bidir, ls_x_bidir,
                     ls_r_bidir, undercoverage_per_shift_bidir, results_ineq_sc_bidir, spread_sc_bidir, load_share_sc_bidir, gini_sc_bidir, results_ineq_perf_bidir,
                     spread_perf_bidir, load_share_perf_bidir, gini_perf_bidir, shift_blocks_bidir) = column_generation_behavior(
                        data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, prob, sp_solver='labeling_bidir'
                    )
                    time_bidir = time.time() - t0_bidir
                    print(f'  --> CG+bidir: obj={final_obj_bidir:.2f}, iter={itr_bidir}, time={time_bidir:.1f}s')

                    # Per-scenario result line
                    speedup = time_mip / time_bidir if time_bidir > 0 else 0
                    obj_match = "YES" if abs(final_obj_behavior - final_obj_bidir) < 0.1 else "NO"
                    print(f'\n  *** SCENARIO {scenario} RESULT ***')
                    print(f'  | CG+Gurobi: obj={final_obj_behavior:.2f}, gap={gap:.2f}%, time={time_mip:.1f}s')
                    print(f'  | CG+Label:  obj={final_obj_bidir:.2f}, gap={gap_bidir:.2f}%, time={time_bidir:.1f}s')
                    print(f'  | Speedup: {speedup:.2f}x, Match: {obj_match}')
                    print()
                    
                    # Store for final summary
                    comparison_results.append({
                        'scenario': scenario,
                        'mip_obj': final_obj_behavior,
                        'mip_time': time_mip,
                        'mip_gap': gap,  # Integrality gap %
                        'bidir_obj': final_obj_bidir,
                        'bidir_time': time_bidir,
                        'bidir_gap': gap_bidir
                    })

                    # Naive (with bidir solver)
                    print('Doing naive with labeling_bidir')
                    (
                    undercoverage_naive, understaffing_naive, perfloss_naive, consistency_naive, consistency_norm_naive,
                    undercoverage_norm_naive, understaffing_norm_naive,
                    perfloss_norm_naive, final_obj_naive, ls_p_naive, ls_sc_naive, ls_perf_naive, ls_x_naive,
                    ls_r_naive, undercoverage_per_shift_naive, results_ineq_sc_naive, spread_sc_naive,
                    load_share_sc_naive, gini_sc_naive, results_ineq_perf_naive,
                    spread_perf_naive, load_share_perf_naive, gini_perf_naive, shift_blocks_naive) = column_generation_naive(data, demand_dict, 0, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, eps, prob, sp_solver='labeling_bidir')


                    shift_undercover_behavior = create_dict_from_list(undercoverage_per_shift_behavior, len(T), len(K))
                    shift_undercover_naive = create_dict_from_list(undercoverage_per_shift_naive, len(T), len(K))


                    daily_undercover_naive = {}
                    for (i, j), value in shift_undercover_naive.items():
                        daily_undercover_naive[i] = daily_undercover_naive.get(i, 0) + value

                    daily_undercover_behavior = {}
                    for (i, j), value in shift_undercover_behavior.items():
                        daily_undercover_behavior[i] = daily_undercover_behavior.get(i, 0) + value


                    # Data frame
                    result = pd.DataFrame([{
                        'I': len(I),
                        'T': len(T),
                        'K': len(K),
                        'pattern': pattern,
                        'scenario': scenario,
                        'prob': prob,
                        'epsilon': eps,
                        'chi': chi,
                        'gap': round(gap, 3),
                        'lagrange': round(lagrangeB, 3),
                        'objval': round(final_obj_behavior, 3),
                        'lbound': round(final_lb, 3),
                        'iteration': itr,
                        'time_sp': round(time_sps, 3),
                        'time_rmp': round(time_rmp, 3),
                        'time_ip': round(time_ip, 3),
                        'undercover_behavior': undercoverage_behavior,
                        'undercover_norm_behavior': undercoverage_norm_behavior,
                        'cons_behavior': consistency_behavior,
                        'cons_norm_behavior': consistency_norm_behavior,
                        'perf_behavior': perfloss_behavior,
                        'perf_norm_behavior': perfloss_norm_behavior,
                        'understaffing_behavior': understaffing_behavior,
                        'understaffing_norm_behavior': understaffing_norm_behavior,
                        'undercover_naive': undercoverage_naive,
                        'undercover_norm_naive': undercoverage_norm_naive,
                        'cons_naive': consistency_naive,
                        'cons_norm_naive': consistency_norm_naive,
                        'perf_naive': perfloss_naive,
                        'perf_norm_naive': perfloss_norm_naive,
                        'understaffing_naive': understaffing_naive,
                        'understaffing_norm_naive': understaffing_norm_naive,
                        'shift_undercover_naive': shift_undercover_naive,
                        'shift_undercover_behavior': shift_undercover_behavior,
                        'perf_list_behavior': ls_perf_behavior,
                        'perf_list_naive': ls_perf_naive,
                        'cons_list_behavior': ls_sc_behavior,
                        'cons_list_naive': ls_sc_naive,
                        'x_list_behavior': ls_x_behavior,
                        'x_list_naive': ls_x_naive,
                        'r_list_behavior': ls_r_behavior,
                        'r_list_naive': ls_r_naive,
                        'p_list_behavior': ls_p_behavior,
                        'p_list_naive': ls_p_naive,
                        'daily_behavior': daily_undercover_behavior,
                        'daily_naive': daily_undercover_naive,
                        'results_ineq_sc_behavior': results_ineq_sc_behavior,
                        'results_ineq_sc_naive': results_ineq_sc_naive,
                        'spread_sc_behavior': spread_sc_behavior,
                        'spread_sc_naive': spread_sc_naive,
                        'load_share_sc_behavior': load_share_sc_behavior,
                        'load_share_sc_naive': load_share_sc_naive,
                        'gini_sc_behavior': gini_sc_behavior,
                        'gini_sc_naive': gini_sc_naive,
                        'results_ineq_perf_behavior': results_ineq_perf_behavior,
                        'results_ineq_perf_naive': results_ineq_perf_naive,
                        'spread_perf_behavior': spread_perf_behavior,
                        'spread_perf_naive': spread_perf_naive,
                        'load_share_perf_behavior': load_share_perf_behavior,
                        'load_share_perf_naive': load_share_perf_naive,
                        'gini_perf_behavior': gini_perf_behavior,
                        'gini_perf_naive': gini_perf_naive,
                        'shift_blocks_behavior': shift_blocks_behavior,
                        'shift_blocks_naive': shift_blocks_naive,
                    }])

                    results = pd.concat([results, result], ignore_index=True)

results.to_excel(f'results/Results_{datetime.now().strftime("%d_%m_%Y_%H-%M")}.xlsx', index=False)

print(results)
print(f"")
print(f"")
print(f"")
print(f"")

print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

# Final Summary Table
print("\n" + "=" * 100)
print("FINAL SUMMARY: CG+MIP vs CG+bidir")
print("=" * 100)
print(f"{'Scen':>6} | {'MIP Obj':>10} | {'MIP Gap':>8} | {'MIP Time':>10} | {'Bidir Obj':>10} | {'Bidir Gap':>8} | {'Bidir Time':>10} | {'Speedup':>8}")
print("-" * 100)
for r in comparison_results:
    speedup = r['mip_time'] / r['bidir_time'] if r['bidir_time'] > 0 else 0
    print(f"{r['scenario']:>6} | {r['mip_obj']:>10.2f} | {r['mip_gap']:>7.2f}% | {r['mip_time']:>9.1f}s | {r['bidir_obj']:>10.2f} | {r['bidir_gap']:>7.2f}% | {r['bidir_time']:>9.1f}s | {speedup:>7.2f}x")
print("-" * 100)

# Averages
if comparison_results:
    avg_mip_time = sum(r['mip_time'] for r in comparison_results) / len(comparison_results)
    avg_mip_gap = sum(r['mip_gap'] for r in comparison_results) / len(comparison_results)
    avg_bidir_time = sum(r['bidir_time'] for r in comparison_results) / len(comparison_results)
    avg_bidir_gap = sum(r['bidir_gap'] for r in comparison_results) / len(comparison_results)
    avg_speedup = avg_mip_time / avg_bidir_time if avg_bidir_time > 0 else 0
    matches = sum(1 for r in comparison_results if abs(r['mip_obj'] - r['bidir_obj']) < 0.1)
    print(f"{'AVG':>6} | {'':>10} | {avg_mip_gap:>7.2f}% | {avg_mip_time:>9.1f}s | {'':>10} | {avg_bidir_gap:>7.2f}% | {avg_bidir_time:>9.1f}s | {avg_speedup:>7.2f}x")
    print(f"\nObj Matches: {matches}/{len(comparison_results)}")
print("=" * 100)