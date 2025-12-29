import gurobipy as gu
import statistics
import numpy as np


class MasterProblem:
    """
    Master Problem for Column Generation using LINEAR constraints.
    
    Performance values are treated as COEFFICIENTS (not variables) in the demand constraints.
    This allows proper column generation where new columns are added with fixed coefficients.
    """
    
    def __init__(self, df, Demand, max_iteration, current_iteration, last, nr, start, start_by_group=None, worker_groups=None):
        self.iteration = current_iteration
        self.max_iteration = max_iteration
        self.nurses = df['I'].dropna().astype(int).unique().tolist()
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.rosterinitial = [1]  # Initial roster has just one element
        self.demand = Demand
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.newvar = {}
        self.last_itr = last
        self.max_itr = max_iteration
        self.cons_lmbda = {}  # Per-group convexity constraints
        self.output_len = nr
        self.demand_values = [self.demand[key] for key in self.demand.keys()]
        self.start = start  # Default start values (backward compatibility)
        self.start_by_group = start_by_group  # Per-group start values
        
        # Worker groups configuration
        self.worker_groups = worker_groups
        if worker_groups is not None:
            # Group-based indexing: group_idx -> (group_name, group_size)
            self.group_info = {}
            for idx, (name, group) in enumerate(worker_groups.items(), start=1):
                self.group_info[idx] = {'name': name, 'size': len(group.worker_ids), 'worker_ids': group.worker_ids}
            self.n_groups = len(self.group_info)
        else:
            # Default: 1 group with all workers
            self.group_info = {1: {'name': 'all', 'size': len(self.nurses), 'worker_ids': self.nurses}}
            self.n_groups = 1
        
        # Track all added schedules: (group_idx, roster_idx) -> schedule dict
        self.all_schedules = {}
        # Track active rosters per group: {group_idx: [roster_indices]}
        self.active_roster_by_group = {g: [1] for g in self.group_info}
        self.active_roster = [1]  # Global roster list

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        # Understaffing variables
        self.u = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='u')
        
        # Lambda variables: lmbda[group_idx, roster_idx]
        # Initially create for all groups with roster index 1
        self.lmbda = {}
        for g in self.group_info:
            for r in self.rosterinitial:
                self.lmbda[g, r] = self.model.addVar(
                    vtype=gu.GRB.CONTINUOUS, lb=0, name=f'lmbda[{g},{r}]'
                )

    def generateConstraints(self):
        # Per-group convexity constraints: sum of lambdas for each group = group_size
        self.cons_lmbda = {}
        for g, info in self.group_info.items():
            self.cons_lmbda[g] = self.model.addConstr(
                gu.quicksum(self.lmbda[g, r] for r in self.rosterinitial) == info['size'],
                name=f"conv_g{g}"
            )
        
        # Demand constraints: LINEAR constraints with performance as coefficients
        for t in self.days:
            for s in self.shifts:
                # Initially, constraint is just: u[t,s] >= demand[t,s]
                # The lmbda terms will be added via setStartSolution and addColumn
                self.cons_demand[t, s] = self.model.addConstr(
                    self.u[t, s] >= self.demand[t, s],
                    name=f"demand({t},{s})"
                )

    def generateObjective(self):
        self.model.setObjective(
            gu.quicksum(self.u[t, s] for t in self.days for s in self.shifts),
            sense=gu.GRB.MINIMIZE
        )

    def getDuals_i(self):
        """Get dual for the per-group convexity constraints."""
        return {g: self.cons_lmbda[g].Pi for g in self.group_info}

    def getDuals_ts(self):
        """Get duals for demand constraints."""
        return {(t, s): self.cons_demand[t, s].Pi for t in self.days for s in self.shifts}

    def updateModel(self):
        self.model.update()

    def setStartSolution(self):
        """Set the initial solution coefficients for roster index 1.
        
        Uses per-group start values if available, otherwise uses default start values.
        The coefficient for each group's lambda contributes group_size to the demand.
        """
        if self.start_by_group is not None:
            # Use per-group start values
            for group_idx, info in self.group_info.items():
                group_name = info['name']
                if group_name in self.start_by_group:
                    perf = self.start_by_group[group_name]['perf']
                    for t in self.days:
                        for s in self.shifts:
                            if (t, s) in perf:
                                # Coefficient = perf * group_size (since λ represents all workers in group)
                                coeff = perf[t, s] * info['size']
                                self.model.chgCoeff(self.cons_demand[t, s], self.lmbda[group_idx, 1], coeff)
        else:
            # Backward compatible: use same start values for all groups
            for group_idx, info in self.group_info.items():
                for t in self.days:
                    for s in self.shifts:
                        if (t, s) in self.start:
                            coeff = self.start[t, s] * info['size']
                            self.model.chgCoeff(self.cons_demand[t, s], self.lmbda[group_idx, 1], coeff)
        self.model.update()

    def addColumn(self, itr, schedule, group_idx=None):
        """
        Add a new column (schedule) to the master problem.
        
        Args:
            itr: Iteration number (column index will be itr + 1)
            schedule: Dictionary {(day, shift, roster_idx): performance_value}
            group_idx: Group index for this column. If None, applies to all groups.
        """
        roster_idx = itr + 1
        
        # Determine which groups this column applies to
        groups = [group_idx] if group_idx is not None else list(self.group_info.keys())
        
        for g in groups:
            group_size = self.group_info[g]['size']
            # Store the schedule
            self.all_schedules[(g, roster_idx)] = schedule
            
            # Build the column coefficients for each constraint
            for t in self.days:
                for s in self.shifts:
                    # Get the performance coefficient for this (day, shift, roster_idx)
                    coeff = schedule.get((t, s, roster_idx), 0.0)
                    if coeff > 0:
                        # Coefficient = perf * group_size
                        self.model.chgCoeff(self.cons_demand[t, s], self.lmbda[g, roster_idx], coeff * group_size)
        
        self.model.update()

    def addLambda(self, itr, group_idx=None):
        """
        Add a new lambda variable for the given iteration.
        
        Args:
            itr: Iteration number (roster index will be itr + 1)
            group_idx: Group index for this lambda. If None, creates for all groups.
        """
        roster_idx = itr + 1
        
        # Determine which groups this lambda applies to
        groups = [group_idx] if group_idx is not None else list(self.group_info.keys())
        
        for g in groups:
            # Create new lambda variable
            self.lmbda[g, roster_idx] = self.model.addVar(
                vtype=gu.GRB.CONTINUOUS, lb=0, name=f'lmbda[{g},{roster_idx}]'
            )
            
            # Track active roster for this group
            if g not in self.active_roster_by_group:
                self.active_roster_by_group[g] = []
            self.active_roster_by_group[g].append(roster_idx)
            
            # Add to per-group convexity constraint
            self.model.chgCoeff(self.cons_lmbda[g], self.lmbda[g, roster_idx], 1.0)
        
        # Track in global active_roster
        if roster_idx not in self.active_roster:
            self.active_roster.append(roster_idx)
        
        self.model.update()

    def printLambdas(self):
        """Get lambda values by roster index (summed across all groups).
        
        Returns dict: {roster_idx: total workers using this roster}
        This maintains backward compatibility with plotPerformanceList.
        """
        vals = {}
        for r in self.active_roster:
            roster_sum = 0
            for g in self.group_info:
                if (g, r) in self.lmbda:
                    roster_sum += round(self.lmbda[g, r].X)
            if roster_sum > 0:
                vals[r] = roster_sum
        return vals

    def finalSolve(self, timeLimit):
        try:
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-2
            self.model.Params.OutputFlag = 1
            
            # Set lambda variables to integer (per-group indexing)
            for g in self.group_info:
                rosters = self.active_roster_by_group.get(g, [1])
                for r in rosters:
                    if (g, r) in self.lmbda:
                        self.lmbda[g, r].VType = gu.GRB.INTEGER
            
            self.model.update()
            self.model.optimize()
            
            if self.model.status == gu.GRB.OPTIMAL:
                print("*" * (self.output_len + 2))
                print("*{:^{output_len}}*".format("***** Integer solution found *****", output_len=self.output_len))
                print("*" * (self.output_len + 2))
            else:
                print("*" * (self.output_len + 2))
                print("*{:^{output_len}}*".format("***** No solution found *****", output_len=self.output_len))
                print("*" * (self.output_len + 2))
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def solveModel(self, timeLimit):
        try:
            self.model.setParam('TimeLimit', timeLimit)
            self.model.Params.OutputFlag = 0
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-7
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-5
            self.model.setParam('ConcurrentMIP', 2)
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def solveRelaxModel(self):
        try:
            self.model.Params.OutputFlag = 0
            self.model.Params.MIPGap = 1e-6
            self.model.Params.Method = 2
            self.model.Params.Crossover = 0
            
            # Ensure all variables are continuous
            for v in self.model.getVars():
                v.VType = gu.GRB.CONTINUOUS
                v.LB = 0.0
            
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def decoy_f(self):
        return None

    def branch_var(self):
        if self.model.status != gu.GRB.OPTIMAL:
            raise Exception("Master problem could not find an optimal solution.")

        lambda_vars = self.model.getVars()

        max_frac = 0
        most_frac_var = None
        for var in lambda_vars:
            try:
                if 'lmbda' in var.VarName:
                    frac = abs(var.X - round(var.X))
                    if frac > max_frac:
                        max_frac = frac
                        most_frac_var = var
            except Exception:
                pass
        return most_frac_var

    def printSolution(self):
        for t in self.days:
            for s in self.shifts:
                print(f"u[{t},{s}] = {self.u[t, s].X}")

    def get_objective(self):
        return self.model.ObjVal

    def calc_behavior(self, lst, ls_sc, scale):
        consistency = sum(ls_sc)
        consistency_norm = sum(ls_sc) / (len(self.nurses) * scale)
        sublist_length = len(lst) // len(self.nurses)
        p_values = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(len(self.nurses))]
        x_values = [[1.0 if value > 0.0 else 0.0 for value in sublist] for sublist in p_values]
        u_results = round(sum(self.u[t, k].X for t in self.days for k in self.shifts), 3)
        sum_xWerte = [sum(row[i] for row in x_values) for i in range(len(x_values[0]))]

        comparison_result = [
            max(0, self.demand_values[i] - sum_xWerte[i])
            for i in range(len(self.demand_values))
        ]

        undercoverage = u_results
        understaffing = round(sum(comparison_result), 5)
        perfloss = round(undercoverage - understaffing, 5)

        undercoverage_norm = undercoverage / (len(self.nurses) * scale)
        understaffing_norm = understaffing / (len(self.nurses) * scale)
        perfloss_norm = perfloss / (len(self.nurses) * scale)

        return undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm

    def calc_naive(self, lst, ls_sc, ls_r, mue, scale):
        consistency = sum(ls_sc)
        perf_ls = []
        consistency_norm = sum(ls_sc) / (len(self.nurses) * scale)
        self.sum_all_doctors = 0
        sublist_length = len(lst) // len(self.nurses)
        sublist_length_short = len(ls_sc) // len(self.nurses)
        p_values = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(len(self.nurses))]
        sc_values2 = [ls_sc[i * sublist_length_short:(i + 1) * sublist_length_short] for i in range(len(self.nurses))]
        r_values2 = [ls_r[i * sublist_length_short:(i + 1) * sublist_length_short] for i in range(len(self.nurses))]
        x_values = [[1.0 if value > 0 else 0.0 for value in sublist] for sublist in p_values]
        u_results = round(sum(self.u[t, k].X for t in self.days for k in self.shifts), 5)
        sum_xWerte = [sum(row[i] for row in x_values) for i in range(len(x_values[0]))]
        self.sum_xWerte = sum_xWerte
        self.sum_all_doctors = 0
        self.sum_values = sum(self.demand_values)
        self.cumulative_sum = [0]
        self.doctors_cumulative_multiplied = []
        self.vals = self.demand_values
        self.comp_result = []
        for i in range(len(self.vals)):
            if self.vals[i] < self.sum_xWerte[i]:
                self.comp_result.append(0)
            else:
                self.comp_result.append(1)
        index = 0
        self.doctors_cumulative_multiplied = []
        cumulative_total = [0] * (len(self.days) * len(self.shifts))
        for i in self.nurses:
            doctor_values = sc_values2[index]
            r_values = r_values2[index]
            x_i_values = x_values[index]
            index += 1
            self.cumulative_sum = [0]
            for i in range(1, len(doctor_values)):
                if r_values[i] == 1 and doctor_values[i] == 0 and self.cumulative_sum[-1] > 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1] - 1)
                elif r_values[i] == 1 and doctor_values[i] == 1 and self.cumulative_sum[-1] > 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1])
                elif r_values[i] == 1 and doctor_values[i] == 0 and self.cumulative_sum[-1] == 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1])
                elif r_values[i] == 1 and doctor_values[i] == 1 and self.cumulative_sum[-1] == 0:
                    self.cumulative_sum.append(self.cumulative_sum[-1])
                else:
                    self.cumulative_sum.append(self.cumulative_sum[-1] + doctor_values[i])
            self.cumulative_sum1 = []
            for element in self.cumulative_sum:
                for _ in range(len(self.shifts)):
                    self.cumulative_sum1.append(element)
            self.cumulative_values = [x * mue for x in self.cumulative_sum1]
            for val in [x * mue for x in self.cumulative_sum]:
                perf_ls.append(round(1 - val, 2))
            self.multiplied_values = [self.cumulative_values[j] * x_i_values[j] for j in range(len(self.cumulative_values))]
            self.multiplied_values1 = [self.multiplied_values[j] * self.comp_result[j] for j in range(len(self.multiplied_values))]
            self.total_sum = sum(self.multiplied_values1)
            self.doctors_cumulative_multiplied.append(self.total_sum)
            self.sum_all_doctors += self.total_sum
            cumulative_total = [cumulative_total[j] + self.multiplied_values1[j] for j in range(len(cumulative_total))]
        undercoverage = u_results + self.sum_all_doctors
        understaffing = u_results
        perfloss = self.sum_all_doctors
        undercoverage_norm = undercoverage / (len(self.nurses) * scale)
        understaffing_norm = understaffing / (len(self.nurses) * scale)
        perfloss_norm = perfloss / (len(self.nurses) * scale)
        return undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm, perf_ls, cumulative_total

    def average_nr_of(self, lst, num_sublists):
        total_length = len(lst)
        sublist_size = total_length // num_sublists
        sublists = [lst[i:i + sublist_size] for i in range(0, total_length, sublist_size)]
        indices_list = []
        for sublist in sublists:
            indices = [index + 1 for index, value in enumerate(sublist) if value == 1.0]
            indices_list.append(indices)
        sums = [sum(sublist) for sublist in sublists]
        mean_value = round(statistics.mean(sums), 5)
        min_value = min(sums)
        max_value = max(sums)
        return sums, mean_value, min_value, max_value, indices_list

    def calculate_variation_coefficient(self, shift_change_days):
        if len(shift_change_days) < 2:
            return 0
        sorted_days = sorted(shift_change_days)
        intervals = np.diff(sorted_days)
        mean = np.mean(intervals)
        std_dev = np.std(intervals)
        variation_coefficient = (std_dev / mean)
        return round(variation_coefficient, 5)

    def gini_coefficient2(self, x):
        x = np.asarray(x)
        if len(x) <= 1:
            return 0
        sorted_x = np.sort(x)
        index = np.arange(1, len(x) + 1)
        n = len(x)
        return ((2 * np.sum(index * sorted_x)) / (n * np.sum(x))) - (n + 1) / n

    def gini_coefficient(self, ls_sc, num_sublists):
        total_length = len(ls_sc)
        sublist_size = total_length // num_sublists
        sublists = [ls_sc[i:i + sublist_size] for i in range(0, total_length, sublist_size)]
        nested = []
        for sublist in sublists:
            cleaned_sublist = [0.0 if x == -0.0 else x for x in sublist]
            nested.append(cleaned_sublist)
        return [self.gini_coefficient2(indices) for indices in nested]

    def compute_autocorrelation_at_lag(self, series, lag):
        n = len(series)
        if lag >= n:
            raise ValueError("Lag is too large for the length of the series.")
        mean = np.mean(series)
        var = np.var(series)
        cov = np.sum((series[:n - lag] - mean) * (series[lag:] - mean)) / n
        autocorrelation = cov / var
        return autocorrelation

    def autoccorrel(self, ls, num_sublists, lags):
        total_length = len(ls)
        sublist_size = total_length // num_sublists
        sublists = [ls[i:i + sublist_size] for i in range(0, total_length, sublist_size)]
        nested = []
        for sublist in sublists:
            cleaned_sublist = [0.0 if x == -0.0 else x for x in sublist]
            nested.append(cleaned_sublist)
        return [self.compute_autocorrelation_at_lag(indices, lags) for indices in nested]

    def getUndercoverage(self):
        return [self.u[t, k].X for t in self.days for k in self.shifts]


# Keep legacy class for backward compatibility if needed
class MasterProblemQC(MasterProblem):
    """Legacy class name for backward compatibility."""
    pass
