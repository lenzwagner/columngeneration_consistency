import gurobipy as gu
import math
import time

class Problem:
    def __init__(self, dfData, DemandDF, eps, Min_WD_i, Max_WD_i, chi):
        self.I = dfData['I'].dropna().astype(int).unique().tolist()
        self.T = dfData['T'].dropna().astype(int).unique().tolist()
        self.K = dfData['K'].dropna().astype(int).unique().tolist()
        self.End = len(self.T)
        self.Week = int(len(self.T) / 7)
        self.Weeks = range(1, self.Week + 1)
        self.demand = DemandDF
        self.model = gu.Model("MasterProblem")
        self.mu = 0.1
        self.epsilon = eps
        self.mue = 0.1
        self.zeta = 0.1
        self.chi = chi
        self.omega = math.floor(1 / 1e-6)
        self.M = len(self.T) + self.omega
        self.xi = 1 - self.epsilon * self.omega
        self.Days_Off = 2
        self.Min_WD = 2
        self.Max_WD = 5
        self.F_S = [(3, 1), (3, 2), (2, 1)]
        self.Days = len(self.T)
        self.demand_values = [self.demand[key] for key in self.demand.keys()]
        self.Min_WD_i = Min_WD_i
        self.Max_WD_i = Max_WD_i

    def buildLinModel(self):
        self.t0 = time.time()
        self.generateVariables()
        self.genGenCons()
        self.genChangesCons()
        self.genRegCons()
        self.model.update()
        self.Recovery()
        self.linPerformance()
        self.generateObjective()
        self.updateModel()


    def generateVariables(self):
        self.x = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="x")
        self.kk = self.model.addVars(self.I, self.Weeks, vtype=gu.GRB.BINARY, name="k")
        self.y = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="y")
        self.o = self.model.addVars(self.T, self.K, vtype=gu.GRB.CONTINUOUS, name="o")
        self.u = self.model.addVars(self.T, self.K, lb = 0, vtype=gu.GRB.CONTINUOUS, name="u")
        self.sc = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="sc")
        self.v = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="v")
        self.q = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="q")
        self.rho = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="rho")
        self.z = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.BINARY, name="z")
        self.perf = self.model.addVars(self.I, self.T, self.K, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="perf")
        self.p = self.model.addVars(self.I, self.T, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="p")
        self.n = self.model.addVars(self.I, self.T, vtype=gu.GRB.INTEGER, ub=self.Days, lb=0, name="n")
        self.n_h = self.model.addVars(self.I, self.T, vtype=gu.GRB.INTEGER, lb=0, ub=self.Days, name="n_h")
        self.h = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="h")
        self.e = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="e")
        self.kappa = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="kappa")
        self.b = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="b")
        self.phi = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="phi")
        self.r = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="r")
        self.f = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="f")
        self.ff = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="ff")
        self.g = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="g")
        self.w = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="w")
        self.gg = self.model.addVars(self.I, self.T, vtype=gu.GRB.CONTINUOUS, lb=-gu.GRB.INFINITY, ub=gu.GRB.INFINITY,
                                     name="gg")
        self.gam = self.model.addVars(self.I, self.T, vtype=gu.GRB.BINARY, name="gam")


    def genGenCons(self):
        for i in self.I:
            for t in self.T:
                self.model.addLConstr(gu.quicksum(self.x[i, t, k] for k in self.K) <= 1)
                self.model.addLConstr(gu.quicksum(self.x[i, t, k] for k in self.K) == self.y[i, t])
        for t in self.T:
            for k in self.K:
                self.model.addLConstr(
                    gu.quicksum(self.perf[i, t, k] for i in self.I) + self.u[t, k] >= self.demand[t, k])
        for i in self.I:
            for t in self.T:
                for k in self.K:
                    self.model.addLConstr(self.x[i, t, k] + self.p[i, t] - 1 <= self.perf[i, t, k])
                    self.model.addLConstr(self.perf[i, t, k] <= self.p[i, t])
                    self.model.addLConstr(self.perf[i, t, k] <= self.x[i, t, k])
        self.model.update()

    def genChangesCons(self):
        for i in self.I:
            self.model.addLConstr(self.gam[i, 1] == 1)
            for t in range(2, len(self.T) + 1):
                self.model.addLConstr(self.gam[i, t] <= self.gam[i, t-1])
                self.model.addLConstr(self.gam[i, t] <= (1 - self.y[i, t - 1]))
                self.model.addLConstr(self.gam[i, t] >= (1 - self.y[i, t -1 ]) + self.gam[i, t-1] - 1)
            for k in self.K:
                for t in self.T:
                    self.model.addLConstr(self.rho[i, t, k] <= 1 - self.q[i, t, k]- self.gam[i, t])
                    self.model.addLConstr(self.rho[i, t, k] <= self.x[i, t, k])
                    self.model.addLConstr(self.rho[i, t, k] >= (1 - self.q[i, t, k]) + self.x[i, t, k] - 1- self.gam[i, t])
                    self.model.addLConstr(self.z[i, t, k] <= self.q[i, t, k])
                    self.model.addLConstr(self.z[i, t, k] <= (1 - self.y[i, t]))
                    self.model.addLConstr(self.z[i, t, k] >= self.q[i, t, k] + (1 - self.y[i, t]) - 1)
                for t in range(1, len(self.T)):
                    self.model.addLConstr(self.q[i, t + 1, k] == self.x[i, t, k] + self.z[i, t, k])
            for t in self.T:
                self.model.addLConstr(1 == gu.quicksum(self.x[i, t, k] for k in self.K) + (1 - self.y[i, t]))
                self.model.addLConstr(gu.quicksum(self.rho[i, t, k] for k in self.K) == self.sc[i, t])
        self.model.update()

    def genRegConsIndivudal(self):
        for i in self.I:
            for t in range(1, len(self.T) - self.Max_WD_i[i] + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max_WD_i[i])) <= self.Max_WD_i[i])
            for t in range(2, len(self.T) - self.Min_WD_i[i] + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t + 1, t + self.Min_WD_i[i] + 1)) >= self.Min_WD_i[i] * (
                                self.y[i, t + 1] - self.y[i, t]))
        for i in self.I:
            for t in range(2, len(self.T) - self.Days_Off + 2):
                for s in range(t + 1, t + self.Days_Off):
                    self.model.addLConstr(1 + self.y[i, t] >= self.y[i, t - 1] + self.y[i, s])
        for i in self.I:
            for k1, k2 in self.F_S:
                for t in range(1, len(self.T)):
                    self.model.addLConstr(self.x[i, t, k1] + self.x[i, t + 1, k2] <= 1)
        self.model.update()

    def genRegCons(self):
        for i in self.I:
            for t in range(1, len(self.T) - self.Max_WD + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t, t + 1 + self.Max_WD)) <= self.Max_WD)
            for t in range(1, len(self.T) - self.Min_WD + 1):
                self.model.addLConstr(
                    gu.quicksum(self.y[i, u] for u in range(t + 1, t + self.Min_WD + 1)) >= self.Min_WD * (
                                self.y[i, t + 1] - self.y[i, t]))
        for i in self.I:
            for t in range(2, len(self.T) - self.Days_Off + 2):
                for s in range(t + 1, t + self.Days_Off):
                    self.model.addLConstr(1 + self.y[i, t] >= self.y[i, t - 1] + self.y[i, s])
        for i in self.I:
            for k1, k2 in self.F_S:
                for t in range(1, len(self.T)):
                    self.model.addLConstr(self.x[i, t, k1] + self.x[i, t + 1, k2] <= 1)
        self.model.update()

    def Recovery(self):
        for i in self.I:
            for t in range(1 + self.chi, len(self.T) + 1):
                self.model.addLConstr(1 <= gu.quicksum(
                    self.sc[i, j] for j in range(t - self.chi, t+1)) + self.r[i, t])
                for k in range(t - self.chi, t+1):
                    self.model.addLConstr(self.sc[i, k] + self.r[i, t] <= 1)
            for t in range(1, 1 + self.chi):
                self.model.addLConstr(0 == self.r[i, t])
        self.model.update()

    def linPerformance(self):
        for i in self.I:
            self.model.addLConstr(0 == self.n[i, 1])
            self.model.addLConstr(0 == self.sc[i, 1])
            self.model.addLConstr(1 == self.p[i, 1])
            self.model.addLConstr(0 == self.h[i, 1])
            for t in self.T:
                self.model.addLConstr(
                    self.omega * self.kappa[i, t] <= gu.quicksum(self.sc[i, j] for j in range(1, t + 1)))
                self.model.addLConstr(gu.quicksum(self.sc[i, j] for j in range(1, t + 1)) <= len(self.T) + (
                        self.omega - 1 - len(self.T)) * (1 - self.kappa[i, t]))
            for t in range(2, len(self.T) + 1):
                self.model.addLConstr(self.ff[i, t] <= self.n[i, t])
                self.model.addLConstr(self.n[i, t] <= len(self.T) * self.ff[i, t])
                self.model.addLConstr(self.b[i, t] <= 1 - self.ff[i, t - 1])
                self.model.addLConstr(self.b[i, t] <= 1 - self.sc[i, t])
                self.model.addLConstr(self.b[i, t] <= self.r[i, t])
                self.model.addLConstr(self.b[i, t] >= self.r[i, t] + (1 - self.ff[i, t - 1]) + (1 - self.sc[i, t]) - 2)
                self.model.addLConstr(self.p[i, t] == 1 - self.epsilon * self.n[i, t] - self.xi * self.kappa[i, t])
                self.model.addLConstr(
                    self.n[i, t] == (self.n[i, t - 1] + self.sc[i, t]) - self.r[i, t] - self.e[i, t] + self.b[i, t])
                self.model.addLConstr(self.omega * self.h[i, t] <= self.n[i, t])
                self.model.addLConstr(self.n[i, t] <= ((self.omega - 1) + self.h[i, t]))
                self.model.addLConstr(self.e[i, t] <= self.sc[i, t])
                self.model.addLConstr(self.e[i, t] <= self.h[i, t - 1])
                self.model.addLConstr(self.e[i, t] >= self.sc[i, t] + self.h[i, t - 1] - 1)
        self.model.update()

    def generateObjective(self):
        self.model.setObjective(gu.quicksum(self.u[t, k] for k in self.K for t in self.T), sense=gu.GRB.MINIMIZE)

    def updateModel(self):
        self.model.update()

    def ModelParams(self):
        self.model.setParam('ConcurrentMIP', 2)

    def solveModel(self):
        self.t1 = time.time()
        try:
            self.model.Params.MIPGap = 0
            self.model.optimize()
        except gu.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
    def get_final_values(self):
        dict = self.model.getAttr("X", self.x)
        liste = list(dict.values())
        final = [0.0 if x == -0.0 else x for x in liste]
        return final

    def setStart(self, start_dict):
        for key, value in start_dict.items():
            self.x[key].Start = value
        self.model.Params.MIPFocus = 3
        self.model.update()

    def getNewSchedule(self):
        return self.model.getAttr("X", self.perf)