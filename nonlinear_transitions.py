import numpy as np

def classify_transition(last_worked, shift, n_shifts=3):
    """
    Classify the transition into a flat index tau for DP.
    tau = last_worked * (n_shifts + 1) + shift
    """
    return int(last_worked * (n_shifts + 1) + shift)

def generate_transitions(eps, chi, omega_max, nl_spec=None, n_shifts=3):
    """
    Precompute the GammaF, GammaH matrices and pi_k array.
    If nl_spec is None, it generates the linear baseline equivalent.
    """
    K_max = omega_max
    H_max = chi + 5 # sufficient upper bound for stability tracking
    
    n_tau = (n_shifts + 1) * (n_shifts + 1)
    
    GammaF = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
    GammaH = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
    pi_k = np.zeros(K_max + 1, dtype=np.float64)
    
    chg_mode = "linear"
    rec_mode = "linear"
    perf_mode = "linear"
    
    delta_matrix = None
    alpha = 1.0
    beta = 1.0
    
    if nl_spec is not None:
        K_max = nl_spec.get("fatigue_levels", K_max)
        H_max = nl_spec.get("stability_levels", H_max)
        GammaF = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
        GammaH = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
        pi_k = np.zeros(K_max + 1, dtype=np.float64)
        
        chg_mode = nl_spec.get("chg_mode", "linear")
        rec_mode = nl_spec.get("rec_mode", "linear")
        perf_mode = nl_spec.get("perf_mode", "linear")
        
        if chg_mode == "matrix":
            delta_matrix = nl_spec.get("delta_matrix")
        
        if rec_mode == "state_dependent":
            alpha = nl_spec.get("alpha", 1.0)
            beta = nl_spec.get("beta", 1.0)
    
    for k in range(K_max + 1):
        for h in range(H_max + 1):
            for tau in range(n_tau):
                last_worked = tau // (n_shifts + 1)
                shift = tau % (n_shifts + 1)
                
                # 1. Determine base change weight
                weight = 0.0
                if chg_mode == "matrix" and delta_matrix is not None:
                    if shift > 0 and last_worked > 0:
                        try:
                            # Use delta matrix. Scale by K_max since fatigue space is integer [0, K_max]
                            # E.g. K_max = 100, 0.13 * 100 = 13 fatigue points.
                            weight = delta_matrix.get(last_worked, {}).get(shift, 0.0) * K_max
                        except Exception:
                            weight = 0.0
                else:
                    # Fallback to simple linear logic if matrix not found or chg_mode != matrix
                    if shift == 0:
                        weight = 0.0
                    elif last_worked == 0:
                        weight = 0.0 # return
                    elif last_worked == shift:
                        weight = 0.0
                    else:
                        weight = 1.0
                    
                # 2. Update H (stability)
                new_h = h + 1
                if shift != 0: # working
                    if weight > 0: # disruption resets stability
                        new_h = 0
                
                new_h = min(new_h, H_max)
                GammaH[k, h, tau] = new_h
                
                # 3. Determine Recovery
                r_new = 0.0
                if rec_mode == "state_dependent":
                    if new_h >= chi + 1:
                        # R(rho) = alpha * (1 - beta^(rho - chi))
                        # r(rho) = R(rho) - R(rho - 1)
                        # = alpha * beta^(rho - chi - 1) * (1 - beta)
                        r_rho = alpha * (beta ** (new_h - chi - 1)) * (1.0 - beta)
                        # Scale to integer grid [0, K_max]
                        r_new = r_rho * K_max
                else:
                    # Old fallback behavior
                    if new_h >= chi + 1:
                        r_new = 1.0
                    if rec_mode == "slow":
                        r_new = 1.0 if new_h >= chi + 2 else 0.0
                
                # 4. Determine new Fatigue
                chg_amt = weight
                if chg_mode == "convex":
                    chg_amt = weight * (1.0 + 0.5 * k)
                elif chg_mode == "concave":
                    chg_amt = weight * (1.0 / (1.0 + 0.5 * k))
                
                new_k = k + chg_amt - r_new
                
                # Bounds
                new_k = max(0, min(int(round(new_k)), K_max))
                GammaF[k, h, tau] = new_k
                
        # Performance
        if rec_mode == "state_dependent":
            p = 1.0 - (k / float(K_max))
        else:
            xi = 1 - eps * omega_max
            if perf_mode == "linear":
                kappa = 1 if k >= omega_max else 0
                p = 1.0 - eps * k - xi * kappa
            elif perf_mode == "threshold":
                if k > K_max / 2:
                    p = 1.0 - eps * k - 0.2
                else:
                    p = 1.0 - eps * k
            else:
                p = 1.0 - eps * k
            
        p = max(0.0, min(p, 1.0))
        pi_k[k] = p
        
    return GammaF, GammaH, pi_k
