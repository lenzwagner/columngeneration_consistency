from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import *
from cg_naive import column_generation_naive
from Utils.Plots.plots import *
from Utils.aggundercover import *
from datetime import *
import time

# DataFrame
results = pd.DataFrame(columns=['I', 'T', 'K', 'pattern', 'scenario', 'prob', 'epsilon', 'chi', 'gap', 'objval', 'lbound', 'iteration', 'time_sp', 'time_rmp',
                                'time_ip', 'undercover_behaviour', 'undercover_norm_behaviour', 'cons_behaviour', 'cons_norm_behaviour', 'perf_behaviour',
                                'perf_norm_behaviour', 'understaffing_behaviour', 'understaffing_norm_behaviour', 'undercover_naive', 'undercover_norm_naive', 'cons_naive',
                                'cons_norm_naive', 'perf_naive', 'perf_norm_naive', 'understaffing_naive', 'understaffing_norm_naive', 'shift_undercover_naive',
                                'shift_undercover_behaviour', 'perf_list_behaviour', 'perf_list_naive', 'cons_list_behaviour', 'cons_list_naive'])

# Times and Parameter
time_Limit, time_cg, time_cg_init, prob = 7200, 7200, 10, 1.0
max_itr, output_len, mue, threshold = 200, 98, 1e-4, 6e-5

start_time = time.time()

# Loop
for epsilon in [0.06]:
    for chi in [3]:
        for len_I in [50,100,150]:
            for pattern in ['Low','Medium','High']:
                for scenario in [1]:
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

                    demand_dict = generate_dict_from_excel('data/demand_scenarios.xlsx', len(I), pattern, scenario=1)
                    df_demand = pd.read_excel('data/data_demand.xlsx', engine='openpyxl')
                    eps = epsilon

                    print(f"")
                    print(f"Iteration: Eps: {epsilon} - Chi: {chi} - I: {len(I)} - Pattern: {pattern} - K: {scenario}")
                    print(f"")

                    ## Column Generation
                    # Bevaior
                    print('Doing behaviour')
                    undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm, final_obj, final_lb, itr, lagrangeB, gap, ls_sc_behav, ls_p_behavior, undercoverage_per_shift, time_sps, time_rmp, time_ip, nr_itr = column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, prob)

                    # Naive
                    print('Doing naive')
                    undercoverage_n, understaffing_n, perfloss_n, consistency_n, consistency_norm_n, undercoverage_norm_n, understaffing_norm_n, perfloss_norm_n, perf_ls_n, cons_ls_n, cumulative_total_n, cumulative_total = column_generation_naive(data, demand_dict, 0, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi,
                                                threshold, time_cg, I, T, K, eps, prob)


                    shift_undercover_naive = create_dict_from_list(cumulative_total, len(T), len(K))
                    shift_undercover_behaviour = create_dict_from_list(undercoverage_per_shift, len(T), len(K))

                    daily_undercover_naive = {}
                    for (i, j), value in shift_undercover_naive.items():
                        daily_undercover_naive[i] = daily_undercover_naive.get(i, 0) + value

                    daily_undercover_behaviour = {}
                    for (i, j), value in shift_undercover_behaviour.items():
                        daily_undercover_behaviour[i] = daily_undercover_behaviour.get(i, 0) + value


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
                        'objval': round(final_obj, 3),
                        'lbound': round(final_lb, 3),
                        'iteration': nr_itr,
                        'time_sp': round(time_sps, 3),
                        'time_rmp': round(time_rmp, 3),
                        'time_ip': round(time_ip, 3),
                        'undercover_behaviour': undercoverage,
                        'undercover_norm_behaviour': undercoverage_norm,
                        'cons_behaviour': consistency,
                        'cons_norm_behaviour': consistency_norm,
                        'perf_behaviour': perfloss,
                        'perf_norm_behaviour': perfloss_norm,
                        'understaffing_behaviour': understaffing,
                        'understaffing_norm_behaviour': understaffing_norm,
                        'undercover_naive': undercoverage_n,
                        'undercover_norm_naive': undercoverage_norm_n,
                        'cons_naive': consistency_n,
                        'cons_norm_naive': consistency_norm_n,
                        'perf_naive': perfloss_n,
                        'perf_norm_naive': perfloss_norm_n,
                        'understaffing_naive': understaffing_n,
                        'understaffing_norm_naive': understaffing_norm_n,
                        'shift_undercover_naive': consistency_n,
                        'shift_undercover_behaviour': consistency_n,
                        'perf_list_behaviour': ls_p_behavior,
                        'perf_list_naive': perf_ls_n,
                        'cons_list_behaviour': ls_sc_behav,
                        'cons_list_naive': cons_ls_n,
                        'daily_behavior': daily_undercover_behaviour,
                        'daily_naive': daily_undercover_naive
                    }])

                    results = pd.concat([results, result], ignore_index=True)

results.to_csv('results/Results.xlsx', index=False)
results.to_excel(f'results/Results_{datetime.now().strftime("%d_%m_%Y_%H-%M")}.xlsx', index=False)

print(f"Total execution time: {time.time() - start_time:.2f} seconds")