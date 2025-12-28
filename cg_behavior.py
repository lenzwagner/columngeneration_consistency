import sys
import time

from masterproblem import *
from subproblem import *
from subproblem_factory import create_subproblem
from Utils.gcutil import *
from Utils.compactsolver import *
from worker_groups import create_homogeneous_group, get_worker_params

def column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi, threshold, time_cg, I, T, K, scale, sp_solver='mip', start_values=None, save_lp=False, worker_groups=None):
    # **** Column Generation ****
    # Prerequisites
    modelImprovable = True
    
    # Backward compatibility: create homogeneous group if none provided
    if worker_groups is None:
        worker_groups = create_homogeneous_group(I, eps, chi)

    # Get Starting Solutions (or use provided ones)
    if start_values is None:
        problem_start = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i, chi, worker_groups=worker_groups)
        problem_start.buildLinModel()
        problem_start.model.Params.MIPFocus = 1
        problem_start.model.Params.Heuristics = 1
        problem_start.model.Params.RINS = 10
        problem_start.model.Params.TimeLimit = time_cg_init
        problem_start.model.update()
        problem_start.model.optimize()

        # Extract starting values per group from representative workers
        # This gives each group a starting column suited to its (epsilon, chi)
        start_values_by_group = {}
        for group_name, group in worker_groups.items():
            rep_worker = group.worker_ids[0]  # Representative worker for this group
            start_values_by_group[group_name] = {
                'perf': {(t, s): problem_start.perf[rep_worker, t, s].x for t in T for s in K},
                'p': {t: problem_start.p[rep_worker, t].x for t in T},
                'x': {(t, s): problem_start.x[rep_worker, t, s].x for t in T for s in K},
                'c': {t: problem_start.sc[rep_worker, t].x for t in T},
                'r': {t: problem_start.r[rep_worker, t].x for t in T},
                'eup': {t: problem_start.e[rep_worker, t].x for t in T},
                'elow': {t: problem_start.b[rep_worker, t].x for t in T},
                'worker_ids': group.worker_ids
            }
        
        # For backward compatibility, use first group's values as default
        first_group = list(start_values_by_group.values())[0]
        start_values_perf = first_group['perf']
        start_values_p = first_group['p']
        start_values_x = first_group['x']
        start_values_c = first_group['c']
        start_values_r = first_group['r']
        start_values_eup = first_group['eup']
        start_values_elow = first_group['elow']
    else:
        # Use provided start values
        start_values_perf = start_values['perf']
        start_values_p = start_values['p']
        start_values_x = start_values['x']
        start_values_c = start_values['c']
        start_values_r = start_values['r']
        start_values_eup = start_values['eup']
        start_values_elow = start_values['elow']
        start_values_by_group = None

    # Initialize iterations
    itr = 0
    last_itr = 0

    # Create empty results lists
    histories = ["objValHistSP", "timeHist", "objValHistRMP", "avg_rc_hist", "lagrange_hist", "sum_rc_hist", "avg_sp_time", "rmp_time_hist", "sp_time_hist"]
    histories_dict = {}
    for history in histories:
        histories_dict[history] = []
    objValHistSP, timeHist, objValHistRMP, avg_rc_hist, lagrange_hist, sum_rc_hist, avg_sp_time, rmp_time_hist, sp_time_hist = histories_dict.values()

    X_schedules = {}
    for index in I:
        X_schedules[f"Physician_{index}"] = []

    Perf_schedules = create_schedule_dict(start_values_perf, 1, T, K)
    Cons_schedules = create_schedule_dict(start_values_c, 1, T)
    Recovery_schedules = create_schedule_dict(start_values_r, 1, T)
    EUp_schedules = create_schedule_dict(start_values_eup, 1, T)
    ELow_schedules = create_schedule_dict(start_values_elow, 1, T)
    P_schedules = create_schedule_dict(start_values_p, 1, T)
    X1_schedules = create_schedule_dict(start_values_x, 1, T, K)

    master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf, 
                           start_by_group=start_values_by_group)
    master.buildModel()

    # Initialize and solve relaxed model
    master.setStartSolution()
    master.updateModel()
    
    # Save initial LP for debugging
    if save_lp:
        import os
        os.makedirs("debug_models", exist_ok=True)
        master.model.write("debug_models/mp_initial.lp")
        print(f"Saved initial LP to debug_models/mp_initial.lp")
    
    master.solveRelaxModel()

    # Retrieve dual values
    duals_i0 = master.getDuals_i()
    duals_ts0 = master.getDuals_ts()

    # Start time count
    t0 = time.time()
    previous_reduced_cost = float('inf')

    while modelImprovable and itr < max_itr:
        print("*{:^{output_len}}*".format(f"Begin Column Generation Iteration {itr}", output_len=output_len))

        # Start
        itr += 1

        # Solve RMP
        rmp_start_time = time.time()
        master.current_iteration = itr + 1
        master.solveRelaxModel()
        rmp_end_time = time.time()
        rmp_time_hist.append(rmp_end_time - rmp_start_time)

        objValHistRMP.append(master.model.objval)
        current_obj = master.model.objval

        # Get and Print Duals
        duals_i = master.getDuals_i()  # Now returns dict: {worker_id: dual}
        duals_ts = master.getDuals_ts()

        # Solve SPs - one per worker group
        modelImprovable = False
        all_reduced_costs = []
        sub_start_time = time.time()
        
        for group_name, group in worker_groups.items():
            # Use representative dual for this group (first worker's dual)
            # In heterogeneous case, we average or use representative
            representative_worker = group.worker_ids[0]
            group_dual_i = duals_i.get(representative_worker, 0.0)
            
            # Build SP with group-specific (epsilon, chi)
            subproblem = create_subproblem(
                sp_solver, group_dual_i, duals_ts, data, representative_worker, itr,
                group.epsilon, Min_WD_i, Max_WD_i, group.chi
            )
            subproblem.buildModel()

            # Solve SP
            if previous_reduced_cost < -0.001:
                subproblem.solveModelNOpt(time_cg)
            else:
                subproblem.solveModelOpt(time_cg)

            # Check if SP is solvable
            status = subproblem.getStatus()
            if status != 2:
                print(f"Warning: Pricing-Problem for group {group_name} not optimal")
                continue

            # Get reduced cost
            reducedCost = subproblem.model.objval
            all_reduced_costs.append(reducedCost)
            print(f'Red. Cost for {group_name}: {reducedCost}')

            # Generate and add columns for each worker in this group
            if reducedCost < -threshold:
                Schedules = subproblem.getNewSchedule()
                
                # Debug: print schedule in first 5 iterations
                if itr <= 5:
                    col_list = sorted([(k[0], k[1]) for k, v in Schedules.items() if v > 0.5])
                    print(f"  [ITR {itr}] [{group_name}] Schedule: {col_list}")
                
                # Add lambda and column for each worker in this group
                for worker_id in group.worker_ids:
                    master.addLambda(itr, worker_id)
                    master.addColumn(itr, Schedules, worker_id)
                
                modelImprovable = True
                
                # Track schedules
                index = representative_worker
                keys = ["X", "Perf", "P", "C", "R", "EUp", "Elow", "X1"]
                methods = ["getOptX", "getOptPerf", "getOptP", "getOptC", "getOptR", "getOptEUp", "getOptElow", "getOptX"]
                schedules = [X_schedules, Perf_schedules, P_schedules, Cons_schedules, Recovery_schedules, EUp_schedules, ELow_schedules, X1_schedules]

                for key, method, schedule in zip(keys, methods, schedules):
                    value = getattr(subproblem, method)()
                    schedule[f"Physician_{index}"].append(value)
        
        sub_end_time = time.time()
        sp_time_hist.append(sub_end_time - sub_start_time)
        timeHist.append(sub_end_time - sub_start_time)

        # Aggregate reduced costs for history
        if all_reduced_costs:
            min_rc = min(all_reduced_costs)
            objValHistSP.append(min_rc)
            previous_reduced_cost = min_rc
        else:
            objValHistSP.append(0.0)
            previous_reduced_cost = 0.0

        # Increase latest used iteration
        last_itr = itr + 1
        master.updateModel()
            
        # Save LP if debugging enabled
        if save_lp:
            import os
            os.makedirs("debug_models", exist_ok=True)
            master.model.write(f"debug_models/mp_{sp_solver}_iter{itr}.lp")

        # Update Model
        master.updateModel()

        # Calculate Metrics
        avg_rc = sum(objValHistSP) / len(objValHistSP)
        lagrange = avg_rc + current_obj
        sum_rc = sum(objValHistSP)
        avg_rc_hist.append(avg_rc)
        sum_rc_hist.append(sum_rc)
        lagrange_hist.append(lagrange)
        objValHistSP.clear()
        avg_time = sum(timeHist) / len(timeHist)
        avg_sp_time.append(avg_time)

        timeHist.clear()

        if not modelImprovable:
            print("*" * (output_len + 2))
            break

    if modelImprovable and itr == max_itr:
        max_itr *= 2

    # Solve Master Problem with integrality restored
    time_ip1 = time.time()
    master.finalSolve(300)
    time_ip2 = time.time() - time_ip1
    objValHistRMP.append(master.model.objval)
    final_obj = master.model.objval
    final_lb = objValHistRMP[-2]

    if abs(final_obj) > 1e-6:
        integrality_gap_pct = ((final_obj - final_lb) / final_obj) * 100
    else:
        integrality_gap_pct = 0.0

    status = master.model.Status
    if status in (gu.GRB.INF_OR_UNBD, gu.GRB.INFEASIBLE, gu.GRB.UNBOUNDED):
        gu.sys.exit(1)

    if status != gu.GRB.OPTIMAL:
        gu.sys.exit(1)

    time_in_sps = sum(avg_sp_time)
    time_in_rmp = time.time()-time_in_sps-time_ip2-t0
    time_in_ip = time_ip2

    objValHistRMP.append(master.model.objval)
    lagranigan_bound = round((objValHistRMP[-2] + sum_rc_hist[-1]), 3)

    ls_p = [round(x, 2) for x in plotPerformanceList(P_schedules, master.printLambdas())]
    ls_sc = [1.0 if x > 0.5 else 0.0 for x in plotPerformanceList(Cons_schedules, master.printLambdas())]
    ls_perf = [round(x, 2) for x in plotPerformanceList(Perf_schedules, master.printLambdas())]
    ls_x = [1.0 if x > 0 else 0.0 for x in ls_perf]
    ls_rec = [1.0 if x > 0.5 else 0.0 for x in plotPerformanceList(Recovery_schedules, master.printLambdas())]

    # Inequality
    L_perf = [x * (1 - p) for x, p in zip(ls_x, ls_perf)]
    results_ineq_sc, spread_sc, load_share_sc, gini_sc = evaluate_inequality(ls_sc, len(master.days), len(master.nurses))
    results_ineq_perf, spread_perf, load_share_perf, gini_perf = evaluate_inequality([sum(L_perf[i:i + 3]) for i in range(0, len(L_perf), 3)], len(master.days), len(master.nurses))

    # shift blocks
    shift_blocks = analyze_and_plot_blocks(ls_x, len(master.nurses), len(master.days), len(master.shifts))

    undercoverage_ab, understaffing_ab, perfloss_ab, consistency_ab, consistency_norm_ab, undercoverage_norm_ab, understaffing_norm_ab, perfloss_norm_ab = master.calc_behavior(ls_perf, ls_sc, scale)

    return round(undercoverage_ab, 5), round(understaffing_ab, 5), round(perfloss_ab, 5), round(consistency_ab, 5), round(consistency_norm_ab, 5), round(undercoverage_norm_ab, 5), round(understaffing_norm_ab, 5), round(perfloss_norm_ab, 5),  round(final_obj, 5), round(final_lb, 5), itr, lagranigan_bound, integrality_gap_pct, time_in_sps, time_in_rmp, time_in_ip, ls_p, ls_sc, ls_perf, ls_x, ls_rec, [0.0 if abs(round(x, 3)) == 0 else round(x, 2) for x in master.getUndercoverage()], results_ineq_sc, spread_sc, load_share_sc, gini_sc, results_ineq_perf, spread_perf, load_share_perf, gini_perf, shift_blocks