import gurobipy as gu
import statistics
import numpy as np


class MasterProblem:
    """
    Master Problem for Column Generation using LINEAR constraints.
    
    Performance values are treated as COEFFICIENTS (not variables) in the demand constraints.
    This allows proper column generation where new columns are added with fixed coefficients.
    """
    
    def __init__(self, df, Demand, max_iteration, current_iteration, last, nr, start):
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
        self.cons_lmbda = None
        self.output_len = nr
        
        # Ensure deterministic order matching sorted(d.keys()) in plotPerformanceList
        self.demand_values = [self.demand[t, s] for t in self.days for s in self.shifts]
        self.start = start
        
        # Track all added schedules (column index -> schedule dict)
        self.all_schedules = {}
        self.active_roster = [1]  # Track which roster indices are active

    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        # Understaffing variables
        self.u = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='u')
        
        # Lambda variables - only create for initial roster
        self.lmbda = {}
        for r in self.rosterinitial:
            self.lmbda[r] = self.model.addVar(vtype=gu.GRB.CONTINUOUS, lb=0, name=f'lmbda[{r}]')

    def generateConstraints(self):
        # Lambda sum constraint: sum of all lambdas = number of nurses
        self.cons_lmbda = self.model.addConstr(
            gu.quicksum(self.lmbda[r] for r in self.rosterinitial) == len(self.nurses),
            name="lmb"
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
        """Get dual for the lambda sum constraint."""
        return self.cons_lmbda.Pi

    def getDuals_ts(self):
        """Get duals for demand constraints."""
        return {(t, s): self.cons_demand[t, s].Pi for t in self.days for s in self.shifts}

    def updateModel(self):
        self.model.update()

    def setStartSolution(self):
        """Set the initial solution coefficients for roster index 1."""
        for t in self.days:
            for s in self.shifts:
                if (t, s) in self.start:
                    # Add the coefficient: performance * lmbda[1]
                    coeff = self.start[t, s]
                    self.model.chgCoeff(self.cons_demand[t, s], self.lmbda[1], coeff)
        self.model.update()

    def addColumn(self, itr, schedule):
        """
        Add a new column (schedule) to the master problem.
        
        Args:
            itr: Iteration number (column index will be itr + 1)
            schedule: Dictionary {(day, shift, roster_idx): performance_value}
        """
        roster_idx = itr + 1
        
        # Store the schedule
        self.all_schedules[roster_idx] = schedule
        
        # Build the column coefficients for each constraint
        for t in self.days:
            for s in self.shifts:
                # Get the performance coefficient for this (day, shift, roster_idx)
                coeff = schedule.get((t, s, roster_idx), 0.0)
                if coeff > 0:
                    # Add coefficient to the demand constraint
                    self.model.chgCoeff(self.cons_demand[t, s], self.lmbda[roster_idx], coeff)
        
        self.model.update()

    def addLambda(self, itr):
        """
        Add a new lambda variable for the given iteration.
        
        Args:
            itr: Iteration number (roster index will be itr + 1)
        """
        roster_idx = itr + 1
        
        # Create new lambda variable
        self.lmbda[roster_idx] = self.model.addVar(
            vtype=gu.GRB.CONTINUOUS, lb=0, name=f'lmbda[{roster_idx}]'
        )
        self.active_roster.append(roster_idx)
        
        # Add to lambda sum constraint
        self.model.chgCoeff(self.cons_lmbda, self.lmbda[roster_idx], 1.0)
        
        self.model.update()

    def printLambdas(self):
        vals = {r: self.lmbda[r].X for r in self.active_roster}
        round_vals = {key: round(value) for key, value in vals.items()}
        return round_vals

    def finalSolve(self, timeLimit):
        try:
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-2
            self.model.Params.OutputFlag = 1
            
            # Set lambda variables to integer
            for r in self.active_roster:
                self.lmbda[r].VType = gu.GRB.INTEGER
            
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
        """
        Calculate behavior metrics.
        
        Key insight: undercoverage = understaffing + perf_loss
        - undercoverage: from u-variables in MP (total shortage)
        - perf_loss: U^Perf = Σ(1-p) for all working shifts (p>0)
        - understaffing: undercoverage - perf_loss
        """
        consistency = sum(ls_sc)
        consistency_norm = sum(ls_sc) / (len(self.nurses) * scale)
        
        # Calculate perf_loss: U^Perf = Σ(1-p) for all p > 0
        # This is the capacity lost due to reduced performance
        perfloss = round(sum(1.0 - p for p in lst if p > 0), 5)
        
        # Undercoverage from u-variables in MP
        undercoverage = round(sum(self.u[t, k].X for t in self.days for k in self.shifts), 3)
        
        # Understaffing = undercoverage - perfloss (can't be negative)
        understaffing = round(max(0, undercoverage - perfloss), 5)

        undercoverage_norm = undercoverage / (len(self.nurses) * scale)
        understaffing_norm = understaffing / (len(self.nurses) * scale)
        perfloss_norm = perfloss / (len(self.nurses) * scale)

        return undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm

    def calc_naive(self, lst, ls_sc, ls_r, mue, scale):
        consistency = sum(ls_sc)
        perf_ls = []
        perf_ls_shift = []
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
        self.sum_values = sum(self.demand_values)
        self.doctors_cumulative_multiplied = []
        self.vals = self.demand_values
        self.comp_result = []
        
        # Safety check to prevent IndexError
        if len(self.vals) != len(self.sum_xWerte):
            print(f"WARNING: Dimension mismatch in calc_naive! Demand: {len(self.vals)}, Assignments: {len(self.sum_xWerte)}")
            # Adjust if possible or pad with zeros to avoid crash
            if len(self.vals) > len(self.sum_xWerte):
                self.sum_xWerte.extend([0] * (len(self.vals) - len(self.sum_xWerte)))
            else:
                self.vals = self.vals + [0] * (len(self.sum_xWerte) - len(self.vals))

        for i in range(len(self.vals)):
            if self.vals[i] < self.sum_xWerte[i]:
                self.comp_result.append(0)
            else:
                self.comp_result.append(1)
        index = 0
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
            
            # Daily performance for ls_p (len = T)
            for val in [x * mue for x in self.cumulative_sum]:
                perf_ls.append(round(1 - val, 2))
                
            # Shift performance for ls_perf (len = T * K)
            self.cumulative_values = [x * mue for x in self.cumulative_sum1]
            for val in self.cumulative_values:
                perf_ls_shift.append(round(1 - val, 2))

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
        return undercoverage, understaffing, perfloss, consistency, consistency_norm, undercoverage_norm, understaffing_norm, perfloss_norm, perf_ls, perf_ls_shift, cumulative_total

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
