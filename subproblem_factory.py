"""
Factory module for creating subproblem solvers.

This module provides a unified interface to create either MIP-based or DP-based
subproblem solvers for the column generation algorithm.
"""

from subproblem import Subproblem
from subproblem_dp import SubproblemDP

# Try to import Numba-optimized version
try:
    from subproblem_dp_optimized import SubproblemDPNumba, SubproblemDPNumba as SubproblemDPBidir, NUMBA_AVAILABLE
except ImportError:
    NUMBA_AVAILABLE = False
    SubproblemDPNumba = None
    SubproblemDPBidir = None


def create_subproblem(solver_type: str, duals_i, duals_ts, df, i, iteration,
                     eps, Min_WD_i, Max_WD_i, chi, use_bidir=False):
    """
    Factory function to create a subproblem solver.

    Args:
        solver_type: 
            - 'mip': Gurobi MIP solver (robust, slower)
            - 'dp': Python DP solver (correct, medium speed)
            - 'labeling': Numba DP solver (forward-only, fast)
            - 'labeling_bidir': Numba DP solver (bidirectional, for T >= 16)
        duals_i: Dual value for lambda constraint
        duals_ts: Dual values for demand constraints
        df: DataFrame with problem data
        i: Staff member index
        iteration: Current CG iteration
        eps: Epsilon parameter for performance degradation
        Min_WD_i: Minimum workdays constraint
        Max_WD_i: Maximum workdays constraint
        chi: Recovery threshold (days without shift change)
        use_bidir: Force bidirectional mode (only for 'labeling' type)

    Returns:
        Subproblem solver instance

    Raises:
        ValueError: If solver_type is not recognized
    """
    solver = solver_type.lower()
    
    if solver == 'mip':
        return Subproblem(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
    
    elif solver == 'dp':
        return SubproblemDP(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
    
    elif solver == 'labeling':
        if not NUMBA_AVAILABLE or SubproblemDPNumba is None:
            print("Warning: Numba not available, falling back to Python DP")
            return SubproblemDP(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
        sp = SubproblemDPNumba(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
        sp._use_bidir = use_bidir  # Flag for bidirectional mode
        return sp
    
    elif solver == 'labeling_bidir':
        if not NUMBA_AVAILABLE or SubproblemDPNumba is None:
            print("Warning: Numba not available, falling back to Python DP")
            return SubproblemDP(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
        sp = SubproblemDPNumba(duals_i, duals_ts, df, i, iteration, eps, Min_WD_i, Max_WD_i, chi)
        sp._use_bidir = True  # Force bidirectional mode
        return sp
    
    else:
        raise ValueError(f"Unknown solver type: {solver_type}. Use 'mip', 'dp', 'labeling', or 'labeling_bidir'.")


