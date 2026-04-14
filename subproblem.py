import gurobipy as gu
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

# =============================================================================
# NUMBA CONFIGURATION
# =============================================================================

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    prange = range

# =============================================================================
# HELPER CLASSES
# =============================================================================

@dataclass
class Label:
    """
    Represents a partial schedule state on day d.

    Attributes:
        day: Current day index
        s_last: Last worked shift type (None if no shift worked yet or on day off)
        last_worked_shift: Most recently worked shift (persists through off days)
        e: Current performance level (0 to floor(1/epsilon))
        rho: Recovery counter (consecutive days without shift change)
        omega: Consecutive workdays counter
        cost: Accumulated reduced cost
        path: List of decisions [(day, shift)] for reconstruction
    """
    day: int
    s_last: Optional[int]  # None means no shift worked yet (or day off)
    last_worked_shift: Optional[int]  # Most recently worked shift (for SC on resume)
    e: int  # Performance level (discretized)
    rho: int  # Recovery counter
    omega: int  # Consecutive workdays
    cost: float
    path: List[Tuple[int, Optional[int]]]  # (day, shift) - None means day off
    total_workdays: int  # Total number of workdays so far

    # Additional tracking for solution reconstruction
    sc_history: List[int]  # Shift change history
    r_history: List[int]   # Recovery history
    p_history: List[float]  # Performance history

    def copy(self):
        """Create a deep copy of this label."""
        return Label(
            day=self.day,
            s_last=self.s_last,
            last_worked_shift=self.last_worked_shift,
            e=self.e,
            rho=self.rho,
            omega=self.omega,
            cost=self.cost,
            path=self.path.copy(),
            total_workdays=self.total_workdays,
            sc_history=self.sc_history.copy(),
            r_history=self.r_history.copy(),
            p_history=self.p_history.copy()
        )

    def dominates(self, other: 'Label', is_final_day: bool = False) -> bool:
        """
        Check if this label dominates the other label.
        
        Conservative dominance: Only dominate if all state variables match
        AND cost is strictly better. This prevents incorrectly pruning
        paths that could lead to better solutions in the future.
        """
        if self is other:
            return True

        if is_final_day:
            return self.cost < other.cost - 1e-9
        else:
            if self.day != other.day: return False
            if self.s_last != other.s_last: return False
            if self.last_worked_shift != other.last_worked_shift: return False
            if self.e != other.e: return False
            if self.rho != other.rho: return False
            if self.omega != other.omega: return False
            return self.cost < other.cost - 1e-9

# =============================================================================
# NUMBA JIT-COMPILED FUNCTIONS
# =============================================================================

@njit(cache=True)
def compute_performance_numba(e: int, omega_max: int, epsilon: float, xi: float) -> float:
    """Compute performance value from state."""
    kappa = 1 if e >= omega_max else 0
    return 1.0 - epsilon * e - xi * kappa


@njit(cache=True)
def pack_state(omega: int, rho: int, e: int, last_worked: int, s_last: int, 
               first_flag: int) -> np.int64:
    """Pack state variables into single int64."""
    return np.int64(((omega + 10) & 0x1F) |
                    ((rho & 0x1F) << 5) |
                    ((e & 0x1F) << 10) |
                    ((last_worked & 0x7) << 15) |
                    ((s_last & 0x7) << 18) |
                    ((first_flag & 0x1) << 21))


@njit(cache=True)
def unpack_state(state: np.int64) -> Tuple[int, int, int, int, int, int]:
    """Unpack state from int64."""
    omega = (state & 0x1F) - 10
    rho = (state >> 5) & 0x1F
    e = (state >> 10) & 0x1F
    last_worked = (state >> 15) & 0x7
    s_last = (state >> 18) & 0x7
    first_flag = (state >> 21) & 0x1
    return omega, rho, e, last_worked, s_last, first_flag


@njit(cache=True)
def forward_pass_numba(
    n_days: int,
    n_shifts: int,
    duals_flat: np.ndarray,
    duals_i: float,
    epsilon: float,
    chi: int,
    omega_max: int,
    xi: float,
    min_wd: int,
    max_wd: int,
    days_off: int,
    suffix_bounds: np.ndarray,
    stop_day: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Forward DP pass from day 0 to stop_day."""
    MAX_STATES = 200000
    
    curr_states = np.zeros(MAX_STATES, dtype=np.int64)
    curr_costs = np.zeros(MAX_STATES, dtype=np.float64)
    curr_paths = np.zeros((MAX_STATES, n_days + 1), dtype=np.int8)
    
    next_states = np.zeros(MAX_STATES, dtype=np.int64)
    next_costs = np.zeros(MAX_STATES, dtype=np.float64)
    next_paths = np.zeros((MAX_STATES, n_days + 1), dtype=np.int8)
    
    # Initial state
    curr_states[0] = pack_state(0, 0, 0, 0, 0, 0)
    curr_costs[0] = -duals_i
    n_curr = 1
    
    best_cost = np.inf
    forbidden = np.array([[3, 1], [3, 2], [2, 1]], dtype=np.int32)
    
    for d in range(stop_day):
        next_day = d + 1
        n_next = 0
        days_remaining = n_days - next_day + 1
        
        for i in range(n_curr):
            state, cost = curr_states[i], curr_costs[i]
            if cost - suffix_bounds[next_day - 1] >= best_cost - 1e-9: continue
            
            omega, rho, e, last_worked, s_last, first_flag = unpack_state(state)
            has_worked = last_worked > 0
            
            # Option 1: Day off
            can_off = True
            if omega > 0 and omega < min_wd:
                if days_remaining >= min_wd or first_flag == 1: can_off = False
            
            if can_off and n_next < MAX_STATES:
                new_omega = -1 if omega > 0 else (omega - 1 if omega < 0 else -1)
                new_rho = rho + 1
                r_new = 1 if new_rho >= chi + 1 else 0
                new_e = max(0, min(e - r_new, omega_max))
                next_states[n_next] = pack_state(new_omega, new_rho, new_e, last_worked, 0, first_flag)
                next_costs[n_next] = cost
                next_paths[n_next, :] = curr_paths[i, :]
                next_paths[n_next, next_day] = -1
                n_next += 1
            
            # Option 2: Work each shift
            for shift in range(1, n_shifts + 1):
                if omega >= max_wd: continue
                if omega < 0 and has_worked and -omega < days_off: continue
                
                is_forbidden = False
                if s_last > 0:
                    for f in range(3):
                        if forbidden[f, 0] == s_last and forbidden[f, 1] == shift:
                            is_forbidden = True; break
                if is_forbidden: continue
                
                c_new = 1 if (last_worked > 0 and last_worked != shift) else 0
                new_omega = 1 if omega <= 0 else omega + 1
                new_rho = rho + 1 if c_new == 0 else 0
                r_new = 1 if new_rho >= chi + 1 else 0
                new_e = max(0, min(e + c_new - r_new, omega_max))
                p_new = compute_performance_numba(new_e, omega_max, epsilon, xi)
                new_cost = cost - duals_flat[(next_day - 1) * n_shifts + (shift - 1)] * p_new
                
                if n_next < MAX_STATES:
                    next_states[n_next] = pack_state(new_omega, new_rho, new_e, shift, shift, 1 if not has_worked and next_day == 1 else first_flag)
                    next_costs[n_next] = new_cost
                    next_paths[n_next, :] = curr_paths[i, :]
                    next_paths[n_next, next_day] = shift
                    n_next += 1
                if next_day == n_days and new_cost < best_cost: best_cost = new_cost
        
        if n_next > 0:
            sort_idx = np.argsort(next_states[:n_next])
            n_pruned, i = 0, 0
            while i < n_next:
                idx = sort_idx[i]
                current_state, best_c = next_states[idx], next_costs[idx]
                j = i + 1
                while j < n_next and next_states[sort_idx[j]] == current_state:
                    if next_costs[sort_idx[j]] < best_c: best_c = next_costs[sort_idx[j]]
                    j += 1
                for k in range(i, j):
                    kdx = sort_idx[k]
                    if abs(next_costs[kdx] - best_c) < 1e-9 and n_pruned < MAX_STATES:
                        curr_states[n_pruned] = current_state
                        curr_costs[n_pruned] = next_costs[kdx]
                        curr_paths[n_pruned, :] = next_paths[kdx, :]
                        n_pruned += 1
                i = j
            n_curr = n_pruned
        else: n_curr = 0
    return curr_states[:n_curr].copy(), curr_costs[:n_curr].copy(), curr_paths[:n_curr].copy(), n_curr


@njit(cache=True)
def forward_pass_from_states(
    n_days: int, n_shifts: int, duals_flat: np.ndarray, epsilon: float, chi: int, omega_max: int, xi: float,
    min_wd: int, max_wd: int, days_off: int, start_day: int, init_states: np.ndarray, init_costs: np.ndarray, n_init: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Forward DP pass from start_day to end, starting from given states."""
    MAX_STATES = 200000
    curr_states, curr_costs, curr_init_idx = np.zeros(MAX_STATES, dtype=np.int64), np.zeros(MAX_STATES, dtype=np.float64), np.zeros(MAX_STATES, dtype=np.int32)
    curr_paths = np.zeros((MAX_STATES, n_days + 1), dtype=np.int8)
    next_states, next_costs, next_init_idx = np.zeros(MAX_STATES, dtype=np.int64), np.zeros(MAX_STATES, dtype=np.float64), np.zeros(MAX_STATES, dtype=np.int32)
    next_paths = np.zeros((MAX_STATES, n_days + 1), dtype=np.int8)
    
    n_curr = min(n_init, MAX_STATES)
    for i in range(n_curr):
        curr_states[i], curr_costs[i], curr_init_idx[i] = init_states[i], init_costs[i], i
    
    forbidden = np.array([[3, 1], [3, 2], [2, 1]], dtype=np.int32)
    for d in range(start_day, n_days):
        next_day, n_next = d + 1, 0
        days_remaining = n_days - next_day + 1
        for i in range(n_curr):
            state, cost, init_idx = curr_states[i], curr_costs[i], curr_init_idx[i]
            omega, rho, e, last_worked, s_last, first_flag = unpack_state(state)
            has_worked = last_worked > 0
            
            if can_off and n_next < MAX_STATES:
                new_omega = -1 if omega > 0 else (omega - 1 if omega < 0 else -1)
                label_rho = rho + 1
                r_new = 1 if label_rho >= chi + 1 else 0
                new_e = max(0, min(e - r_new, omega_max))
                next_states[n_next] = pack_state(new_omega, rho + 1, new_e, last_worked, 0, first_flag)
                next_costs[n_next], next_init_idx[n_next] = cost, init_idx
                next_paths[n_next, :], next_paths[n_next, next_day] = curr_paths[i, :], -1
                n_next += 1
            
            for shift in range(1, n_shifts + 1):
                if omega >= max_wd or (omega < 0 and has_worked and -omega < days_off): continue
                is_forbidden = False
                if s_last > 0:
                    for f in range(3):
                        if forbidden[f, 0] == s_last and forbidden[f, 1] == shift: is_forbidden = True; break
                if is_forbidden: continue
                c_new = 1 if (last_worked > 0 and last_worked != shift) else 0
                label_rho = rho + 1 if c_new == 0 else 0
                r_new = 1 if label_rho >= chi + 1 else 0
                new_e = max(0, min(e + c_new - r_new, omega_max))
                p_new = compute_performance_numba(new_e, omega_max, epsilon, xi)
                if n_next < MAX_STATES:
                    next_states[n_next] = pack_state(1 if omega <= 0 else omega + 1, label_rho, new_e, shift, shift, 1 if not has_worked and next_day == start_day + 1 else first_flag)
                    next_costs[n_next] = cost - duals_flat[(next_day - 1) * n_shifts + (shift - 1)] * p_new
                    next_init_idx[n_next], next_paths[n_next, :], next_paths[n_next, next_day] = init_idx, curr_paths[i, :], shift
                    n_next += 1
        
        if n_next > 0:
            sort_idx = np.argsort(next_states[:n_next])
            n_pruned, i = 0, 0
            while i < n_next:
                idx = sort_idx[i]
                current_state, best_c, best_idx = next_states[idx], next_costs[idx], idx
                j = i + 1
                while j < n_next and next_states[sort_idx[j]] == current_state:
                    if next_costs[sort_idx[j]] < best_c: best_c, best_idx = next_costs[sort_idx[j]], sort_idx[j]
                    j += 1
                curr_states[n_pruned], curr_costs[n_pruned], curr_init_idx[n_pruned] = current_state, best_c, next_init_idx[best_idx]
                curr_paths[n_pruned, :] = next_paths[best_idx, :]
                n_pruned += 1
                i = j
            n_curr = n_pruned
        else: n_curr = 0
    return curr_states[:n_curr], curr_costs[:n_curr], curr_init_idx[:n_curr], curr_paths[:n_curr], n_curr


@njit(cache=True)
def merge_bidirectional(
    fwd_states: np.ndarray, fwd_costs: np.ndarray, fwd_paths: np.ndarray, n_fwd: int,
    second_costs: np.ndarray, second_init_idx: np.ndarray, second_paths: np.ndarray, n_second: int,
    n_days: int, mid_day: int,
) -> Tuple[float, np.ndarray]:
    """Merge first and second forward passes."""
    best_cost, best_path = np.inf, np.zeros(n_days + 1, dtype=np.int8)
    for i in range(n_second):
        if second_costs[i] < best_cost:
            best_cost, init_idx = second_costs[i], second_init_idx[i]
            for d in range(mid_day + 1): best_path[d] = fwd_paths[init_idx, d]
            for d in range(mid_day + 1, n_days + 1): best_path[d] = second_paths[i, d]
    return best_cost, best_path

# =============================================================================
# SOLVER CLASSES
# =============================================================================

class Subproblem:
    """MIP-based subproblem solver."""
    def __init__(self, duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi):
        self.days = df['T'].dropna().astype(int).unique().tolist()
        self.shifts = df['K'].dropna().astype(int).unique().tolist()
        self.duals_i, self.duals_ts = duals_i, duals_ts
        self.model = gu.Model("Subproblem")
        self.index, self.itr = i, iteration + 1
        self.epsilon, self.chi = eps, chi
        self.omega = math.floor(1 / (self.epsilon + 1e-6))
        self.xi = 1 - self.epsilon * self.omega
        self.Min_WD, self.Max_WD, self.Days_Off = 2, 5, 2
        self.F_S = [(3, 1), (3, 2), (2, 1)]

    def buildModel(self):
        self.x = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.BINARY, name="x")
        self.y = self.model.addVars(self.days, vtype=gu.GRB.BINARY, name="y")
        self.sc = self.model.addVars(self.days, vtype=gu.GRB.BINARY, name="sc")
        self.performance = self.model.addVars(self.days, self.shifts, [self.itr], vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="performance")
        self.p = self.model.addVars(self.days, vtype=gu.GRB.CONTINUOUS, lb=0, ub=1, name="p")
        self.n = self.model.addVars(self.days, vtype=gu.GRB.INTEGER, ub=len(self.days), lb=0, name="n")
        self.h, self.e, self.kappa, self.b, self.ff, self.gam, self.r = [self.model.addVars(self.days, vtype=gu.GRB.BINARY) for _ in range(7)]
        self.rho = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.BINARY)
        self.q = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.BINARY)
        self.z = self.model.addVars(self.days, self.shifts, vtype=gu.GRB.BINARY)

        for t in self.days:
            self.model.addLConstr(gu.quicksum(self.x[t, k] for k in self.shifts) == self.y[t])
            for k in self.shifts:
                self.model.addLConstr(self.performance[t, k, self.itr] >= self.p[t] + self.x[t, k] - 1)
                self.model.addLConstr(self.performance[t, k, self.itr] <= self.p[t])
                self.model.addLConstr(self.performance[t, k, self.itr] <= self.x[t, k])
        
        self.model.addLConstr(self.gam[1] == 1)
        for t in range(2, len(self.days) + 1):
            self.model.addLConstr(self.gam[t] <= self.gam[t - 1])
            self.model.addLConstr(self.gam[t] <= (1 - self.y[t - 1]))
            self.model.addLConstr(self.gam[t] >= (1 - self.y[t - 1]) + self.gam[t - 1] - 1)
        for k in self.shifts:
            for t in self.days:
                self.model.addLConstr(self.rho[t, k] <= 1 - self.q[t, k] - self.gam[t])
                self.model.addLConstr(self.rho[t, k] <= self.x[t, k])
                self.model.addLConstr(self.rho[t, k] >= (1 - self.q[t, k]) + self.x[t, k] - 1 - self.gam[t])
                self.model.addLConstr(self.z[t, k] <= self.q[t, k])
                self.model.addLConstr(self.z[t, k] <= (1 - self.y[t]))
                self.model.addLConstr(self.z[t, k] >= self.q[t, k] + (1 - self.y[t]) - 1)
            for t in range(1, len(self.days)):
                self.model.addLConstr(self.q[t + 1, k] == self.x[t, k] + self.z[t, k])

        for t in self.days:
            self.model.addLConstr(gu.quicksum(self.rho[t, k] for k in self.shifts) == self.sc[t])
        for t in range(2, len(self.days) - self.Days_Off + 2):
            for s in range(t + 1, min(t + self.Days_Off, len(self.days) + 1)):
                self.model.addLConstr(1 + self.y[t] >= self.y[t - 1] + self.y[s])
        for k1, k2 in self.F_S:
            if k1 in self.shifts and k2 in self.shifts:
                for t in range(1, len(self.days)): self.model.addLConstr(self.x[t, k1] + self.x[t + 1, k2] <= 1)
        
        for t in range(1 + self.chi, len(self.days) + 1):
            self.model.addLConstr(1 <= gu.quicksum(self.sc[j] for j in range(t - self.chi, t+1)) + self.r[t])
            for k in range(t - self.chi, t + 1): self.model.addLConstr(self.sc[k] + self.r[t] <= 1)
        for t in range(1, 1 + self.chi): self.model.addLConstr(0 == self.r[t])
        
        self.model.addLConstr(0 == self.n[1]); self.model.addLConstr(0 == self.sc[1])
        self.model.addLConstr(1 == self.p[1]); self.model.addLConstr(0 == self.h[1])
        for t in self.days:
            self.model.addLConstr(self.omega * self.kappa[t] <= gu.quicksum(self.sc[j] for j in range(1, t + 1)))
            self.model.addLConstr(gu.quicksum(self.sc[j] for j in range(1, t + 1)) <= len(self.days) + (self.omega - 1 - len(self.days)) * (1 - self.kappa[t]))
        for t in range(2, len(self.days) + 1):
            self.model.addLConstr(self.ff[t] <= self.n[t]); self.model.addLConstr(self.n[t] <= len(self.days) * self.ff[t])
            self.model.addLConstr(self.b[t] <= 1 - self.ff[t-1]); self.model.addLConstr(self.b[t] <= 1 - self.sc[t])
            self.model.addLConstr(self.b[t] <= self.r[t]); self.model.addLConstr(self.b[t] >= self.r[t] + (1 - self.ff[t-1]) + (1 - self.sc[t]) - 2)
            self.model.addLConstr(self.p[t] == 1 - self.epsilon * self.n[t] - self.xi * self.kappa[t])
            self.model.addLConstr(self.n[t] == (self.n[t - 1] + self.sc[t])-self.r[t]-self.e[t]+self.b[t])
            self.model.addLConstr(self.omega * self.h[t] <= self.n[t]); self.model.addLConstr(self.n[t] <= ((self.omega - 1) + self.h[t]))
            self.model.addLConstr(self.e[t] <= self.sc[t]); self.model.addLConstr(self.e[t] <= self.h[t - 1])
            self.model.addLConstr(self.e[t] >= self.sc[t] + self.h[t - 1] - 1)
        
        for t in range(1, len(self.days) - self.Max_WD + 1): self.model.addLConstr(gu.quicksum(self.y[u] for u in range(t, t + 1 + self.Max_WD)) <= self.Max_WD)
        for t in range(1, len(self.days) - self.Min_WD + 1): self.model.addLConstr(gu.quicksum(self.y[u] for u in range(t + 1, t + self.Min_WD + 1)) >= self.Min_WD * (self.y[t + 1] - self.y[t]))
        if len(self.days) >= self.Min_WD: self.model.addLConstr(gu.quicksum(self.y[u] for u in range(1, 1 + self.Min_WD)) >= self.Min_WD * self.y[1])
        
        self.model.setObjective(0 - gu.quicksum(self.performance[t, s, self.itr] * self.duals_ts[t, s] for t in self.days for s in self.shifts) - self.duals_i, sense=gu.GRB.MINIMIZE)
        self.model.update()

    def solveModelOpt(self, timeLimit):
        self.model.Params.TimeLimit, self.model.Params.Threads, self.model.Params.OutputFlag, self.model.Params.MIPGap = timeLimit, 0, 0, 0.00001
        self.model.optimize()
    def solveModelNOpt(self, timeLimit):
        self.model.Params.TimeLimit, self.model.Params.Threads, self.model.Params.OutputFlag, self.model.Params.MIPGap = timeLimit, 0, 0, 0.05
        self.model.optimize()
    def getStatus(self): return self.model.status
    def getNewSchedule(self): return self.model.getAttr("X", self.performance)
    def getOptX(self): return self.model.getAttr("X", self.x)
    def getOptP(self): return self.model.getAttr("X", self.p)
    def getOptC(self): return self.model.getAttr("X", self.sc)
    def getOptR(self): return self.model.getAttr("X", self.r)
    def getOptEUp(self): return self.model.getAttr("X", self.e)
    def getOptElow(self): return self.model.getAttr("X", self.b)
    def getOptPerf(self): return self.model.getAttr("X", self.performance)

class SubproblemDP:
    """Base DP labeling solver."""
    def __init__(self, duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi):
        self.itr, self.days, self.shifts = iteration + 1, df['T'].dropna().astype(int).unique().tolist(), df['K'].dropna().astype(int).unique().tolist()
        self.duals_i, self.duals_ts, self.epsilon, self.chi = duals_i, duals_ts, eps, chi
        self.omega_max = math.floor(1 / (eps + 1e-6)); self.xi = 1 - eps * self.omega_max
        self.Min_WD, self.Max_WD, self.Days_Off, self.F_S = 2, 5, 2, [(3, 1), (3, 2), (2, 1)]
        self.status, self.objval, self.best_label = None, None, None
        self.model = type('Mock', (), {'objval': property(lambda s: self.objval)})()
    def buildModel(self): pass
    def solveModelOpt(self, timeLimit): self._solve(timeLimit)
    def solveModelNOpt(self, timeLimit): self._solve(timeLimit)
    def _solve(self, timeLimit):
        labels = {0: [Label(0, None, None, 0, 0, 0, -self.duals_i, [], 0, [], [], [])]}
        for d in [0] + self.days[:-1]:
            labels[d+1] = []
            for l in labels[d]:
                if (not (l.omega > 0 and l.omega < self.Min_WD and (len(self.days)-d >= self.Min_WD or (any(s is not None for d_, s in l.path) and l.path[0][1] is not None)))):
                    nl = l.copy(); nl.day, nl.s_last, nl.omega = d+1, None, (-1 if l.omega > 0 else l.omega-1)
                    nl.rho = l.rho + 1
                    r_new = 1 if nl.rho >= self.chi+1 else 0
                    nl.e = max(0, min(l.e-r_new, self.omega_max))
                    nl.cost, nl.path = l.cost, l.path + [(d+1, None)]
                    nl.sc_history, nl.r_history, nl.p_history = l.sc_history + [0], l.r_history + [r_new], l.p_history + [1.0 - self.epsilon*nl.e - self.xi*(1 if nl.e>=self.omega_max else 0)]
                    labels[d+1].append(nl)
                for s in self.shifts:
                    if l.omega < self.Max_WD and (not (l.omega < 0 and any(s_ is not None for d_, s_ in l.path) and abs(l.omega) < self.Days_Off)) and (not any(l.s_last == f1 and s == f2 for f1, f2 in self.F_S)):
                        nl = l.copy(); nl.day, nl.total_workdays, nl.s_last, nl.last_worked_shift = d+1, l.total_workdays+1, s, s
                        c_new = 1 if l.last_worked_shift is not None and l.last_worked_shift != s else 0
                        nl.omega, nl.rho = (1 if l.omega < 0 else l.omega+1), (l.rho+1 if c_new==0 else 0)
                        r_new = 1 if nl.rho >= self.chi+1 else 0
                        nl.e = max(0, min(l.e+c_new-r_new, self.omega_max))
                        p_new = 1.0 - self.epsilon*nl.e - self.xi*(1 if nl.e>=self.omega_max else 0)
                        nl.cost, nl.path = l.cost - self.duals_ts.get((d+1, s), 0.0)*p_new, l.path + [(d+1, s)]
                        nl.sc_history, nl.r_history, nl.p_history = l.sc_history + [c_new], l.r_history + [r_new], l.p_history + [p_new]
                        labels[d+1].append(nl)
            best_per_state = {}
            for l in labels[d+1]:
                k = (l.day, l.s_last, l.last_worked_shift, l.e, l.rho, l.omega)
                if k not in best_per_state or l.cost < best_per_state[k].cost: best_per_state[k] = l
            labels[d+1] = list(best_per_state.values())
        if not labels[self.days[-1]]: self.status = gu.GRB.INFEASIBLE; return
        self.best_label = min(labels[self.days[-1]], key=lambda l: l.cost); self.status, self.objval = gu.GRB.OPTIMAL, self.best_label.cost
    def getStatus(self): return self.status if self.status else gu.GRB.LOADED
    def getNewSchedule(self):
        if not self.best_label: return {}
        return {(d, s, self.itr): (self.best_label.p_history[d-1] if self.best_label.path[d-1][1]==s else 0.0) for d in self.days for s in self.shifts}
    def getOptX(self): return {(d, s): (1.0 if self.best_label.path[d-1][1]==s else 0.0) for d in self.days for s in self.shifts} if self.best_label else {}
    def getOptP(self): return {d: self.best_label.p_history[d-1] for d in self.days} if self.best_label else {}
    def getOptC(self): return {d: float(self.best_label.sc_history[d-1]) for d in self.days} if self.best_label else {}
    def getOptR(self): return {d: float(self.best_label.r_history[d-1]) for d in self.days} if self.best_label else {}
    def getOptEUp(self): return {d: 0.0 for d in self.days}
    def getOptElow(self): return {d: 1.0 if self.best_label.r_history[i]==1 and self.best_label.sc_history[i]==0 else 0.0 for i, d in enumerate(self.days)} if self.best_label else {}
    def getOptPerf(self): return self.getNewSchedule()

class SubproblemDPOptimized(SubproblemDP):
    """Alias for base DP with prune-on-fly, kept for compatibility."""
    pass

class SubproblemDPNumba:
    """Numba-optimized DP solver."""
    def __init__(self, duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi):
        self.itr, self.days, self.shifts = iteration + 1, df['T'].dropna().astype(int).unique().tolist(), df['K'].dropna().astype(int).unique().tolist()
        self.duals_i, self.duals_ts, self.epsilon, self.chi = duals_i, duals_ts, eps, chi
        self.omega_max = math.floor(1 / (eps + 1e-6)); self.xi = 1 - eps * self.omega_max
        self.Min_WD, self.Max_WD, self.Days_Off = 2, 5, 2
        self.status, self.objval, self.best_path, self.all_optimal_paths = None, None, None, []
        self.model = type('Mock', (), {'objval': property(lambda s: self.objval)})()
        n_days, n_shifts = len(self.days), len(self.shifts)
        self.duals_flat = np.zeros(n_days * n_shifts)
        for d_idx, day in enumerate(self.days):
            for k_idx, shift in enumerate(self.shifts): self.duals_flat[d_idx * n_shifts + k_idx] = duals_ts.get((day, shift), 0.0)
        self.suffix_bounds = np.zeros(n_days + 2)
        for d_idx in range(n_days-1, -1, -1): self.suffix_bounds[d_idx] = self.suffix_bounds[d_idx+1] + max(self.duals_flat[d_idx*n_shifts:(d_idx+1)*n_shifts])
    def buildModel(self): pass
    def solveModelOpt(self, timeLimit):
        n_days, n_shifts = len(self.days), len(self.shifts)
        if not NUMBA_AVAILABLE: return SubproblemDP.solveModelOpt(self, timeLimit)
        if getattr(self, '_use_bidir', False) and n_days >= 8:
            mid = n_days // 2
            fs, fc, fp, nf = forward_pass_numba(n_days, n_shifts, self.duals_flat, self.duals_i, self.epsilon, self.chi, self.omega_max, self.xi, self.Min_WD, self.Max_WD, self.Days_Off, self.suffix_bounds, mid)
            if nf > 0:
                _, sc, si, sp, ns = forward_pass_from_states(n_days, n_shifts, self.duals_flat, self.epsilon, self.chi, self.omega_max, self.xi, self.Min_WD, self.Max_WD, self.Days_Off, mid, fs, fc, nf)
                if ns > 0:
                    self.objval, self.best_path = merge_bidirectional(fs, fc, fp, nf, sc, si, sp, ns, n_days, mid)
                    self.all_optimal_paths, self.status = [self.best_path.copy()], gu.GRB.OPTIMAL; return
        states, costs, paths, n_s = forward_pass_numba(n_days, n_shifts, self.duals_flat, self.duals_i, self.epsilon, self.chi, self.omega_max, self.xi, self.Min_WD, self.Max_WD, self.Days_Off, self.suffix_bounds, n_days)
        if n_s > 0:
            self.objval = np.min(costs[:n_s]); self.all_optimal_paths = [paths[i].copy() for i in range(n_s) if abs(costs[i]-self.objval)<1e-9]
            self.best_path, self.status = self.all_optimal_paths[0], gu.GRB.OPTIMAL
        else: self.objval, self.status = float('inf'), gu.GRB.INFEASIBLE
    def solveModelNOpt(self, tL): self.solveModelOpt(tL)
    def getStatus(self): return self.status if self.status else gu.GRB.LOADED
    def getNewSchedule(self):
        if self.best_path is None: return {}
        p_v = self.getOptP()
        return {(day, shift, self.itr): (p_v.get(day, 1.0) if self.best_path[d_idx+1]==shift else 0.0) for d_idx, day in enumerate(self.days) for shift in self.shifts}
    def getOptX(self): return {(day, shift): (1.0 if self.best_path[d_idx+1]==shift else 0.0) for d_idx, day in enumerate(self.days) for shift in self.shifts} if self.best_path is not None else {}
    def _reconstruct(self):
        if self.best_path is None: return None
        n_days, last, c_h, r_h = len(self.days), 0, [], []
        for d in range(n_days):
            s = int(self.best_path[d+1])
            c = (1 if last>0 and last!=s else 0) if s>0 else 0
            if s>0: last = s
            c_h.append(c)
        for d in range(n_days): r_h.append((1 if sum(c_h[max(0, d-self.chi):d+1])==0 else 0) if d>=self.chi else 0)
        e, p_h, eu_h, el_h = 0, [], [], []
        for d in range(n_days):
            cl, rl = c_h[d], r_h[d]
            el = (1 if e==0 else 0)*(1-cl)*rl; eu = cl*(1 if e>=self.omega_max else 0)
            e = max(0, min(e+cl+el-rl-eu, self.omega_max)); p_h.append(1.0-self.epsilon*e)
            eu_h.append(float(eu)); el_h.append(float(el))
        return {'p': p_h, 'sc': c_h, 'r': r_h, 'e_up': eu_h, 'e_low': el_h}
    def getOptP(self): h = self._reconstruct(); return {d: h['p'][i] for i, d in enumerate(self.days)} if h else {}
    def getOptC(self): h = self._reconstruct(); return {d: float(h['sc'][i]) for i, d in enumerate(self.days)} if h else {}
    def getOptR(self): h = self._reconstruct(); return {d: float(h['r'][i]) for i, d in enumerate(self.days)} if h else {}
    def getOptEUp(self): h = self._reconstruct(); return {d: h['e_up'][i] for i, d in enumerate(self.days)} if h else {}
    def getOptElow(self): h = self._reconstruct(); return {d: h['e_low'][i] for i, d in enumerate(self.days)} if h else {}
    def getOptPerf(self): return self.getNewSchedule()
    def getOptimalCount(self): return len(self.all_optimal_paths)

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_subproblem(solver_type: str, duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi, use_bidir=False):
    """Consolidated factory to create subproblem solvers."""
    solver = solver_type.lower()
    if solver == 'mip': return Subproblem(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
    if solver == 'dp': return SubproblemDP(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
    if solver in ['labeling', 'labeling_bidir']:
        if not NUMBA_AVAILABLE: return SubproblemDP(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
        sp = SubproblemDPNumba(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
        sp._use_bidir = (solver == 'labeling_bidir' or use_bidir); return sp
    raise ValueError(f"Unknown solver: {solver_type}")