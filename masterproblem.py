import gurobipy as gu
import statistics
import numpy as np
class MasterProblem:
    def __init__(self, df, Demand, max_iteration, current_iteration, last, nr, start):
        self.iteration = current_iteration
        self.max_iteration = max_iteration
        self.nurses = df['I'].dropna().astype(int).unique().tolist()
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self._current_iteration = current_iteration
        self.roster = [i for i in range(1, self.max_iteration + 2)]
        self.rosterinitial = [i for i in range(1, 2)]
        self.demand = Demand
        self.model = gu.Model("MasterProblem")
        self.cons_demand = {}
        self.newvar = {}
        self.last_itr = last
        self.max_itr = max_iteration
        self.cons_lmbda = {}
        self.output_len = nr
        self.demand_values = [self.demand[key] for key in self.demand.keys()]
        self.start = start


    def buildModel(self):
        self.generateVariables()
        self.generateConstraints()
        self.model.update()
        self.generateObjective()
        self.model.update()

    def generateVariables(self):
        self.u = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.CONTINUOUS, lb=0, name='u')
        self.performance_i = self.model.addVars(self.days, self.shifts, self.roster,
                                               vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name='performance_i')
        self.lmbda = self.model.addVars(self.roster, vtype=gu.GRB.INTEGER, lb=0, name='lmbda')

    def generateConstraints(self):
        self.cons_lmbda = self.model.addLConstr(len(self.nurses) == gu.quicksum(self.lmbda[r] for r in self.rosterinitial), name = "lmb")
        for t in self.days:
            for s in self.shifts:
                self.cons_demand[t, s] = self.model.addConstr(
                    gu.quicksum(self.performance_i[t, s, r] * self.lmbda[r] for r in self.rosterinitial) +
                    self.u[t, s] >= self.demand[t, s], "demand("+str(t)+","+str(s)+")")
        return self.cons_lmbda, self.cons_demand

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.u[t, s] for t in self.days for s in self.shifts),
                                sense=gu.GRB.MINIMIZE)

    def getDuals_i(self):
        Pi_cons_lmbda = self.cons_lmbda.Pi
        return Pi_cons_lmbda

    def getDuals_ts(self):
        Pi_cons_demand = self.model.getAttr("QCPi", self.cons_demand)
        return Pi_cons_demand

    def updateModel(self):
        self.model.update()

    def setStartSolution(self):
        for t in self.days:
            for s in self.shifts:
                if (t, s) in self.start:
                    self.model.addLConstr(self.performance_i[t, s, 1] == self.start[t, s])

    def addColumn(self, itr, schedule):
        self.rosterIndex = itr + 1
        for t in self.days:
            for s in self.shifts:
                qexpr = self.model.getQCRow(self.cons_demand[t, s])
                qexpr.add(schedule[t, s, self.rosterIndex] * self.lmbda[self.rosterIndex], 1)
                rhs = self.cons_demand[t, s].getAttr('QCRHS')
                sense = self.cons_demand[t, s].getAttr('QCSense')
                name = self.cons_demand[t, s].getAttr('QCName')
                newcon = self.model.addQConstr(qexpr, sense, rhs, name)
                self.model.remove(self.cons_demand[t, s])
                self.cons_demand[t, s] = newcon
        self.model.update()

    def printLambdas(self):
        vals = self.model.getAttr("X", self.lmbda)

        round_vals = {key: round(value) for key, value in vals.items()}

        return round_vals

    def addLambda(self, itr):
        self.rosterIndex = itr + 1
        self.newlmbcoef = 1.0
        current_lmb_cons = self.cons_lmbda
        expr = self.model.getRow(current_lmb_cons)
        new_lmbcoef = self.newlmbcoef
        expr.add(self.lmbda[self.rosterIndex], new_lmbcoef)
        rhs_lmb = current_lmb_cons.getAttr('RHS')
        sense_lmb = current_lmb_cons.getAttr('Sense')
        name_lmb = current_lmb_cons.getAttr('ConstrName')
        newconlmb = self.model.addLConstr(expr, sense_lmb, rhs_lmb, name_lmb)
        self.model.remove(current_lmb_cons)
        self.cons_lmbda = newconlmb

    def finalSolve(self, timeLimit):
        try:
            self.model.Params.IntegralityFocus = 1
            self.model.Params.FeasibilityTol = 1e-9
            self.model.Params.BarConvTol = 0.0
            self.model.Params.MIPGap = 1e-2
            self.model.Params.OutputFlag = 1
            self.model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
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
            self.model.Params.QCPDual = 1
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
            self.model.Params.QCPDual = 1
            for v in self.model.getVars():
                v.setAttr('vtype', 'C')
                v.setAttr('lb', 0.0)
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

    def decoy_f(self):
        return None
    def branch_var(self):

        if self.model.status != gu.GRB.OPTIMAL:
            raise Exception("Master problem could not find an optimal solution.")

        lambda_vars = self.model.getVars()
        fractional_values = {}

        for var in lambda_vars:
            if 'lmbda' in var.varName:
                value = var.x
                if abs(value - round(value)) > 1e-6:
                    fractional_values[var.varName] = value

        if not fractional_values:
            print("No fractional lambdas.")
            return None, None

        most_fractional_var = max(
            fractional_values.items(),
            key=lambda x: abs(x[1] - round(x[1]))
        )
        var_name, var_value = most_fractional_var

        print(f'most Frac: {most_fractional_var}')

        i, r = map(int, var_name.split('_')[1:])

        print(f"Most fractional variable: lambda_{i}_{r} with value {var_value}")
        return (i, r), var_value

    def calc_behavior(self, lst, ls_sc, scale):
        consistency = sum(ls_sc)
        consistency_norm = sum(ls_sc) / (len(self.nurses)*scale)
        sublist_length = len(lst) // len(self.nurses)
        p_values = [lst[i * sublist_length:(i + 1) * sublist_length] for i in range(len(self.nurses))]

        x_values = [[1.0 if value > 0.0 else 0.0 for value in sublist] for sublist in p_values]
        u_results = round(sum(self.u[t, k].x for t in self.days for k in self.shifts), 3)
        sum_xWerte = [sum(row[i] for row in x_values) for i in range(len(x_values[0]))]


        comparison_result = [
            max(0, self.demand_values[i] - sum_xWerte[i])
            for i in range(len(self.demand_values))
        ]

        undercoverage = u_results
        understaffing = round(sum(comparison_result), 5)
        perfloss = round(undercoverage - understaffing, 5)

        ##print(f"Undercoverage before: {undercoverage}")
        #print(f"Understaffing before: {understaffing}")
        #print(f"PerformanceLo before: {perfloss}")


        # Noramlized Values
        undercoverage_norm = undercoverage / (len(self.nurses)*scale)
        understaffing_norm = understaffing / (len(self.nurses)*scale)
        perfloss_norm = perfloss / (len(self.nurses)*scale)

        # Output
        #print(
            #"\nUndercoverage: {:.4f}\nUnderstaffing: {:.4f}\nPerformance Loss: {:.4f}\nConsistency: {:.4f}\nNorm_Undercoverage: {:.4f}\nNorm_Understaffing: {:.4f}\nNorm_Performance Loss: {:.4f}\nNorm_Consistency: {:.4f}\n".format(
               # undercoverage,
               # understaffing, perfloss, consistency, undercoverage_norm, understaffing_norm, perfloss_norm,
                #consistency_norm))

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

        u_results = round(sum(self.u[t, k].x for t in self.days for k in self.shifts), 5)
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

        # Initialize the list with 28*3 zeros
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

            self.multiplied_values = [self.cumulative_values[j] * x_i_values[j] for j in
                                      range(len(self.cumulative_values))]
            self.multiplied_values1 = [self.multiplied_values[j] * self.comp_result[j] for j in
                                       range(len(self.multiplied_values))]
            self.total_sum = sum(self.multiplied_values1)
            self.doctors_cumulative_multiplied.append(self.total_sum)
            self.sum_all_doctors += self.total_sum

            print('Test', len(cumulative_total),cumulative_total, len(self.multiplied_values1), self.multiplied_values1, sep ="\n")
            cumulative_total = [cumulative_total[j] + self.multiplied_values1[j] for j in range(len(cumulative_total))]

        undercoverage = u_results + self.sum_all_doctors
        understaffing = u_results
        perfloss = self.sum_all_doctors

        # Normalized values
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
        #print(variation_coefficient)
        #print(shift_change_days)

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
        """

        Args:
        series (list or np.array): Row.
        lag (int): Lag.

        Returns:
        float: The autocorrelation value for the given lag.
        """
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
        return [self.u[t, k].x for t in self.days for k in self.shifts]

