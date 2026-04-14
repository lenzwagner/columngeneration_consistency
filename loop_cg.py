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
                                'shift_undercover_behavior', 'daily_undercover_behavior', 'daily_undercover_naive', 'perf_list_behavior', 'perf_list_naive', 'cons_list_behavior', 'cons_list_naive', 'p_list_behavior', 'p_list_naive',
                                'spread_sc_naive', 'load_share_sc_behavior', 'load_share_sc_naive', 'gini_sc_behavior', 'gini_sc_naive', 'top10_sc_behavior', 'top10_sc_naive', 'results_ineq_perf_behavior',
                                'results_ineq_perf_naive', 'spread_perf_behavior', 'spread_perf_naive', 'load_share_perf_behavior', 'load_share_perf_naive', 'gini_perf_behavior',
                                'gini_perf_naive', 'top10_perf_behavior', 'top10_perf_naive', 'shift_blocks_behavior', 'shift_blocks_naive', 'changes_sequence'])

# Times and Parameter
time_Limit, time_cg, time_cg_init, prob = 7200, 7200, 20, 1.0
max_itr, output_len, mue, threshold = 200, 98, 1e-4, 6e-5

start_time = time.time()

# Loop
for epsilon in [0.06]:
    for chi in [5]:
        for len_I in [50,100,150]:
            for pattern in ['Low', 'Medium', 'High']:
                for scenario in range(1, 26):
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

                    data = build_data_frame(I, T, K)

                    random.seed = 2
                    demand_dict = generate_dict_from_excel('data/demand_data.xlsx', len(I), pattern, scenario=scenario)

                    eps = epsilon

                    print(f"")
                    print(f"Iteration: Eps: {epsilon} - Chi: {chi} - I: {len(I)} - Pattern: {pattern} - K: {scenario}")
                    print(f"")

                    ## Column Generation - CG+bidir (Labeling subproblem)
                    print('Doing behaviour with CG+Labeling (bidir SP)')
                    t0_bidir = time.time()
                    (undercoverage_behavior, understaffing_behavior, perfloss_behavior, consistency_behavior, 
                     consistency_norm_behavior, undercoverage_norm_behavior, understaffing_norm_behavior,
                     perfloss_norm_behavior, final_obj_behavior, final_lb, itr, lagrangeB, gap, time_sps, time_rmp, 
                     time_ip, ls_p_behavior, ls_sc_behavior, ls_perf_behavior, ls_x_behavior,
                     ls_r_behavior, undercoverage_per_shift_behavior, results_ineq_sc_behavior, spread_sc_behavior, 
                     load_share_sc_behavior, gini_sc_behavior, top10_sc_behavior, results_ineq_perf_behavior,
                     spread_perf_behavior, load_share_perf_behavior, gini_perf_behavior, top10_perf_behavior, 
                     shift_blocks_behavior) = column_generation_behavior(
                        data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, prob, sp_solver='labeling_bidir', save_lp=True
                    )
                    time_bidir = time.time() - t0_bidir
                    print(f'  --> CG+bidir: obj={final_obj_behavior:.2f}, iter={itr}, gap={gap:.2f}%, time={time_bidir:.1f}s')

                    # Naive (with bidir solver)
                    print('Doing naive with labeling_bidir')
                    (
                    undercoverage_naive, understaffing_naive, perfloss_naive, consistency_naive, consistency_norm_naive,
                    undercoverage_norm_naive, understaffing_norm_naive,
                    perfloss_norm_naive, final_obj_naive, ls_p_naive, ls_sc_naive, ls_perf_naive, ls_x_naive,
                    ls_r_naive, undercoverage_per_shift_naive, results_ineq_sc_naive, spread_sc_naive,
                    load_share_sc_naive, gini_sc_naive, top10_sc_naive, results_ineq_perf_naive,
                    spread_perf_naive, load_share_perf_naive, gini_perf_naive, top10_perf_naive, shift_blocks_naive) = column_generation_naive(data, demand_dict, 0, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, eps, prob, sp_solver='labeling_bidir')


                    shift_undercover_behavior = create_dict_from_list(undercoverage_per_shift_behavior, len(T), len(K))
                    shift_undercover_naive = create_dict_from_list(undercoverage_per_shift_naive, len(T), len(K))


                    daily_undercover_naive = dict_reducer(shift_undercover_naive)
                    daily_undercover_behavior = dict_reducer(shift_undercover_behavior)


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
                        'time_total': round(time_bidir, 3),
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
                        'daily_undercover_behavior': daily_undercover_behavior,
                        'daily_undercover_naive': daily_undercover_naive,
                        'results_ineq_sc_behavior': results_ineq_sc_behavior,
                        'results_ineq_sc_naive': results_ineq_sc_naive,
                        'spread_sc_behavior': spread_sc_behavior,
                        'spread_sc_naive': spread_sc_naive,
                        'load_share_sc_behavior': load_share_sc_behavior,
                        'load_share_sc_naive': load_share_sc_naive,
                        'gini_sc_behavior': gini_sc_behavior,
                        'gini_sc_naive': gini_sc_naive,
                        'top10_sc_behavior': top10_sc_behavior,
                        'top10_sc_naive': top10_sc_naive,
                        'results_ineq_perf_behavior': results_ineq_perf_behavior,
                        'results_ineq_perf_naive': results_ineq_perf_naive,
                        'spread_perf_behavior': spread_perf_behavior,
                        'spread_perf_naive': spread_perf_naive,
                        'load_share_perf_behavior': load_share_perf_behavior,
                        'load_share_perf_naive': load_share_perf_naive,
                        'gini_perf_behavior': gini_perf_behavior,
                        'gini_perf_naive': gini_perf_naive,
                        'top10_perf_behavior': top10_perf_behavior,
                        'top10_perf_naive': top10_perf_naive,
                        'shift_blocks_behavior': shift_blocks_behavior,
                        'shift_blocks_naive': shift_blocks_naive,
                        'changes_sequence': combine_lists(ls_x_behavior, ls_x_naive, len(T), len(I)),
                    }])

                    results = pd.concat([results, result], ignore_index=True)

results.to_excel(f'results/Test_{datetime.now().strftime("%d_%m_%Y_%H-%M")}.xlsx', index=False)

print(results)
print(f"")

print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: CG+bidir Results")
print("=" * 80)
for idx, row in results.iterrows():
    print(f"Scenario {row['scenario']}: obj={row['objval']:.2f}, LB={row['lbound']:.2f}, gap={row['gap']:.2f}%, iter={row['iteration']}, time={row['time_total']:.1f}s")
print("=" * 80)