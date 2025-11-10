from masterproblem import *
from subproblem import *
from Utils.gcutil import *
from Utils.compactsolver import *

def column_generation_behavior(data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, output_len, chi, threshold, time_cg, I, T, K, scale):
    # **** Column Generation ****
    # Prerequisites
    modelImprovable = True

    # Get Starting Solutions
    problem_start = Problem(data, demand_dict, eps, Min_WD_i, Max_WD_i, chi)
    problem_start.buildLinModel()
    problem_start.model.Params.MIPFocus = 1
    problem_start.model.Params.Heuristics = 1
    problem_start.model.Params.RINS = 10
    problem_start.model.Params.TimeLimit = time_cg_init
    problem_start.model.update()
    problem_start.model.optimize()

    # Schedules
    # Create
    start_values_perf = {(t, s): problem_start.perf[1, t, s].x for t in T for s in K}
    start_values_p = {(t): problem_start.p[1, t].x for t in T}
    start_values_x = {(t, s): problem_start.x[1, t, s].x for t in T for s in K}
    start_values_c = {(t): problem_start.sc[1, t].x for t in T}
    start_values_r = {(t): problem_start.r[1, t].x for t in T}
    start_values_eup = {(t): problem_start.e[1, t].x for t in T}
    start_values_elow = {(t): problem_start.b[1, t].x for t in T}

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

    master = MasterProblem(data, demand_dict, max_itr, itr, last_itr, output_len, start_values_perf)
    master.buildModel()

    # Initialize and solve relaxed model
    master.setStartSolution()
    master.updateModel()
    master.solveRelaxModel()

    # Retrieve dual values
    duals_i0 = master.getDuals_i()
    duals_ts0 = master.getDuals_ts()
    print(f"{duals_i0, duals_ts0}")

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
        duals_i = master.getDuals_i()
        duals_ts = master.getDuals_ts()

        # Solve SPs
        modelImprovable = False

        # Build SP
        subproblem = Subproblem(duals_i, duals_ts, data, 1, itr, eps, Min_WD_i, Max_WD_i, chi)
        subproblem.buildModel()

        # Save time to solve SP
        sub_start_time = time.time()
        if previous_reduced_cost < -0.001:
            print("*{:^{output_len}}*".format(f"Use MIP-Gap > 0 in Iteration {itr}", output_len=output_len))
            subproblem.solveModelNOpt(time_cg)
        else:
            print("*{:^{output_len}}*".format(f"Use MIP-Gap = 0 in Iteration {itr}", output_len=output_len))
            subproblem.solveModelOpt(time_cg)
        sub_end_time = time.time()
        sp_time_hist.append(sub_end_time - sub_start_time)

        sub_totaltime = sub_end_time - sub_start_time
        timeHist.append(sub_totaltime)
        index = 1

        keys = ["X", "Perf", "P", "C", "R", "EUp", "Elow", "X1"]
        methods = ["getOptX", "getOptPerf", "getOptP", "getOptC", "getOptR", "getOptEUp", "getOptElow", "getOptX"]
        schedules = [X_schedules, Perf_schedules, P_schedules, Cons_schedules, Recovery_schedules, EUp_schedules, ELow_schedules, X1_schedules]

        for key, method, schedule in zip(keys, methods, schedules):
            value = getattr(subproblem, method)()
            schedule[f"Physician_{index}"].append(value)

        # Check if SP is solvable
        status = subproblem.getStatus()
        if status != 2:
            raise Exception("*{:^{output_len}}*".format("Pricing-Problem can not reach optimality!", output_len=output_len))

        # Save ObjVal History
        reducedCost = subproblem.model.objval
        objValHistSP.append(reducedCost)

        # Update previous_reduced_cost for the next iteration
        previous_reduced_cost = reducedCost
        print("*{:^{output_len}}*".format(f"Reduced Costs in Iteration {itr}: {reducedCost}", output_len=output_len))

        # Increase latest used iteration
        last_itr = itr + 1

        # Generate and add columns with reduced cost
        if reducedCost < -threshold:
            Schedules = subproblem.getNewSchedule()
            master.addColumn(itr, Schedules)
            master.addLambda(itr)
            master.updateModel()
            modelImprovable = True

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
    master.finalSolve(300)
    objValHistRMP.append(master.model.objval)
    final_obj = master.model.objval
    final_lb = objValHistRMP[-2]

    status = master.model.Status
    if status in (gu.GRB.INF_OR_UNBD, gu.GRB.INFEASIBLE, gu.GRB.UNBOUNDED):
        #print("The model cannot be solved because it is infeasible or unbounded")
        gu.sys.exit(1)

    if status != gu.GRB.OPTIMAL:
        #print(f"Optimization was stopped with status {status}")
        gu.sys.exit(1)

    nSolutions = master.model.SolCount
    #print(f"Number of solutions found: {nSolutions}")

    # Print objective values of solutions
    for e in range(nSolutions):
        master.model.setParam(gu.GRB.Param.SolutionNumber, e)
        #print(f"{master.model.PoolObjVal:g} ", end="")
        if e % 15 == 14:
            print("")
    #print("")

    objValHistRMP.append(master.model.objval)

    lagranigan_bound = round((objValHistRMP[-2] + sum_rc_hist[-1]), 3)

    # Calc Stats
    undercoverage_pool = []
    understaffing_pool = []
    perf_pool = []
    cons_pool = []
    undercoverage_pool_norm = []
    understaffing_pool_norm = []
    perf_pool_norm = []
    cons_pool_norm = []

    sol = master.printLambdas()

    ls_p_d = plotPerformanceList(P_schedules, sol)

    ls_sc = plotPerformanceList(Cons_schedules, sol)
    ls_p = plotPerformanceList(Perf_schedules, sol)
    ls_r = process_recovery(ls_sc, chi, len(T))

    undercoverage_ab, understaffing_ab, perfloss_ab, consistency_ab, consistency_norm_ab, undercoverage_norm_ab, understaffing_norm_ab, perfloss_norm_ab = master.calc_behavior(
        ls_p, ls_sc, scale)

    undercoverage_pool.append(undercoverage_ab)
    understaffing_pool.append(understaffing_ab)
    perf_pool.append(perfloss_ab)
    cons_pool.append(consistency_ab)
    undercoverage_pool_norm.append(undercoverage_norm_ab)
    understaffing_pool_norm.append(understaffing_norm_ab)
    perf_pool_norm.append(perfloss_norm_ab)
    cons_pool_norm.append(consistency_norm_ab)


    #print(f"Total feasible solutions processed: {len(undercoverage_pool)}")
    #print(f"Under-List: {undercoverage_pool}")
    #print(f"Perf-List: {perf_pool}")
    #print(f"Cons-List: {cons_pool}")

    undercoverage = min(undercoverage_pool)
    understaffing = min(understaffing_pool)
    perfloss = min(perf_pool)
    consistency = min(cons_pool)
    undercoverage_norm = min(undercoverage_pool_norm)
    understaffing_norm = min(understaffing_pool_norm)
    perfloss_norm = min(perf_pool_norm)
    consistency_norm = min(cons_pool_norm)
    # Coefficients
    sums, mean_value, min_value, max_value, indices_list = master.average_nr_of(ls_sc, len(master.nurses))
    variation_coefficients = [master.calculate_variation_coefficient(indices) for indices in indices_list]
    mean_variation_coefficient = (round(np.mean(variation_coefficients) * 100, 4))
    min_variation_coefficient = (round(np.min(variation_coefficients) * 100, 4))
    max_variation_coefficient = (round(np.max(variation_coefficients) * 100, 4))
    std_variation_coefficient = (round(np.std(variation_coefficients) * 100, 4))
    results_sc = [mean_variation_coefficient, min_variation_coefficient, max_variation_coefficient, std_variation_coefficient]

    sums_r, mean_value_r, min_value_r, max_value_r, indices_list_r = master.average_nr_of(ls_r, len(master.nurses))
    variation_coefficients_r = [master.calculate_variation_coefficient(indices) for indices in indices_list_r]
    mean_variation_coefficient_r = (round(np.mean(variation_coefficients_r) * 100, 4))
    min_variation_coefficient_r = (round(np.min(variation_coefficients_r) * 100, 4))
    max_variation_coefficient_r = (round(np.max(variation_coefficients_r) * 100, 4))
    std_variation_coefficient_r = (round(np.std(variation_coefficients_r)* 100, 4))
    results_r = [mean_variation_coefficient_r, min_variation_coefficient_r, max_variation_coefficient_r, std_variation_coefficient_r]

    undercoverage_behavior = master.getUndercoverage()
    # Gini
    #gini_sc = master.gini_coefficient(ls_sc, len(master.nurses))
    #gini_r = master.gini_coefficient(ls_r, len(master.nurses))

    autocorrel = master.autoccorrel(ls_sc, len(master.nurses), 2)

    return round(undercoverage, 5), round(understaffing, 5), round(perfloss, 5), round(consistency, 5), round(consistency_norm, 5), round(undercoverage_norm, 5), round(understaffing_norm, 5), round(perfloss_norm, 5), results_sc, results_r, autocorrel, round(final_obj, 5), round(final_lb, 5), itr, lagranigan_bound, ls_sc, ls_p_d, undercoverage_behavior