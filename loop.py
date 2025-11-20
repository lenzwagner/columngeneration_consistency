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

# Loop
for epsilon in [0.06]:
    for chi in [3]:
        for len_I in [7]:
            for pattern in ['Low']:
                for scenario in range(1, 2):
                    if pattern == 'Medium':
                        prob = 1.0
                    elif pattern == 'High':
                        prob = 1.1
                    elif pattern == 'Low':
                        prob = 0.9

                    # Data
                    T = list(range(1, 8))
                    I = list(range(1, len_I + 1))
                    K = [1, 2, 3]

                    data = pd.DataFrame({
                        'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
                        'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
                        'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
                    })

                    random.seed = 2
                    demand_dict2 = generate_dict_from_excel('data/demand_scenarios.xlsx', len(I), pattern, scenario)
                    demand_dict = {(1, 1): 3, (1, 2): 1, (1, 3): 3, (2, 1): 2, (2, 2): 2, (2, 3): 1, (3, 1): 1, (3, 2): 2, (3, 3): 1, (4, 1): 3, (4, 2): 1, (4, 3): 2, (5, 1): 1, (5, 2): 3, (5, 3): 2, (6, 1): 1, (6, 2): 4, (6, 3): 1, (7, 1): 1, (7, 2): 3, (7, 3): 2}
                    print(demand_dict)
                    eps = epsilon

                    print(f"")
                    print(f"Iteration: Eps: {epsilon} - Chi: {chi} - I: {len(I)} - Pattern: {pattern} - K: {scenario}")
                    print(f"")

                    ## Column Generation
                    # Bevaior
                    print('Doing behaviour')

                    (undercoverage_behavior, understaffing_behavior, perfloss_behavior, consistency_behavior, consistency_norm_behavior, undercoverage_norm_behavior, understaffing_norm_behavior,
                     perfloss_norm_behavior, final_obj_behavior, final_lb, itr, lagrangeB, gap, time_sps, time_rmp, time_ip, ls_p_behavior, ls_sc_behavior, ls_perf_behavior, ls_x_behavior,
                     ls_r_behavior, undercoverage_per_shift_behavior, results_ineq_sc_behavior, spread_sc_behavior, load_share_sc_behavior, gini_sc_behavior, results_ineq_perf_behavior,
                     spread_perf_behavior, load_share_perf_behavior, gini_perf_behavior, shift_blocks_behavior) = column_generation_behavior(
                        data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, prob
                    )



                    # Naive
                    print('Doing naive')
                    (
                    undercoverage_naive, understaffing_naive, perfloss_naive, consistency_naive, consistency_norm_naive,
                    undercoverage_norm_naive, understaffing_norm_naive,
                    perfloss_norm_naive, final_obj_naive, ls_p_naive, ls_sc_naive, ls_perf_naive, ls_x_naive,
                    ls_r_naive, undercoverage_per_shift_naive, results_ineq_sc_naive, spread_sc_naive,
                    load_share_sc_naive, gini_sc_naive, results_ineq_perf_naive,
                    spread_perf_naive, load_share_perf_naive, gini_perf_naive, shift_blocks_naive) = column_generation_naive(data, demand_dict, 0, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, eps, prob)


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

print(f"Total execution time: {time.time() - start_time:.2f} seconds")