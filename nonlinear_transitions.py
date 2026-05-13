import numpy as np

def classify_transition(last_worked, shift, n_shifts=3):
    """
    Classify the transition into a flat index tau for DP.
    tau = last_worked * (n_shifts + 1) + shift
    """
    return int(last_worked * (n_shifts + 1) + shift)

def generate_transitions(eps, chi, omega_max, nl_spec=None, n_shifts=3):
    """
    Precompute the GammaF, GammaH, GammaNu matrices and pi_k array.
    If nl_spec is None, it generates the linear baseline equivalent.
    """
    K_max = omega_max
    H_max = chi + 5 # sufficient upper bound for stability tracking
    Nu_max = 28 # sufficient upper bound for change-sequence length tracking
    
    n_tau = (n_shifts + 1) * (n_shifts + 1)
    
    chg_mode = "linear"
    rec_mode = "linear"
    perf_mode = "linear"
    
    delta_matrix = None
    alpha_R = 1.0
    gamma_R = 1.0
    gamma_C = 1.0
    e_max = 1.0
    
    if nl_spec is not None:
        K_max = nl_spec.get("fatigue_levels", K_max)
        H_max = nl_spec.get("stability_levels", H_max)
        
        chg_mode = nl_spec.get("chg_mode", "linear")
        rec_mode = nl_spec.get("rec_mode", "linear")
        perf_mode = nl_spec.get("perf_mode", "linear")
        
        if chg_mode == "matrix":
            delta_matrix = nl_spec.get("delta_matrix")
        
        if rec_mode == "state_dependent":
            alpha_R = nl_spec.get("alpha_R", 0.04)
            gamma_R = nl_spec.get("gamma_R", 0.5)
            gamma_C = nl_spec.get("gamma_C", 1.25)
            e_max = nl_spec.get("e_max", 1.0)
            
    GammaF = np.zeros((K_max + 1, H_max + 1, Nu_max + 1, n_tau), dtype=np.int32)
    GammaH = np.zeros((H_max + 1, n_tau), dtype=np.int32)
    GammaNu = np.zeros((Nu_max + 1, n_tau), dtype=np.int32)
    pi_k = np.zeros(K_max + 1, dtype=np.float64)

    # Precompute GammaH and GammaNu since they don't depend on k
    for h in range(H_max + 1):
        for tau in range(n_tau):
            last_worked = tau // (n_shifts + 1)
            shift = tau % (n_shifts + 1)
            
            weight = 0.0
            if chg_mode == "matrix" and delta_matrix is not None:
                if shift > 0 and last_worked > 0:
                    weight = delta_matrix.get(last_worked, {}).get(shift, 0.0)
            else:
                if shift == 0 or last_worked == 0 or last_worked == shift:
                    weight = 0.0
                else:
                    weight = 1.0
            
            new_h = h + 1
            if shift != 0:
                if weight > 0: # disruption resets stability
                    new_h = 0
            
            GammaH[h, tau] = min(new_h, H_max)

    for nu in range(Nu_max + 1):
        for tau in range(n_tau):
            last_worked = tau // (n_shifts + 1)
            shift = tau % (n_shifts + 1)
            
            weight = 0.0
            if chg_mode == "matrix" and delta_matrix is not None:
                if shift > 0 and last_worked > 0:
                    weight = delta_matrix.get(last_worked, {}).get(shift, 0.0)
            else:
                if shift == 0 or last_worked == 0 or last_worked == shift:
                    weight = 0.0
                else:
                    weight = 1.0
                    
            if shift != 0 and weight > 0:
                new_nu = nu + 1
            else:
                new_nu = 0
                
            GammaNu[nu, tau] = min(new_nu, Nu_max)

    e_max_int = int(e_max * K_max)

    for k in range(K_max + 1):
        for h in range(H_max + 1):
            for nu in range(Nu_max + 1):
                for tau in range(n_tau):
                    last_worked = tau // (n_shifts + 1)
                    shift = tau % (n_shifts + 1)
                    
                    new_h = GammaH[h, tau]
                    new_nu = GammaNu[nu, tau]
                    
                    weight = 0.0
                    if chg_mode == "matrix" and delta_matrix is not None:
                        if shift > 0 and last_worked > 0:
                            weight = delta_matrix.get(last_worked, {}).get(shift, 0.0)
                    else:
                        if shift == 0 or last_worked == 0 or last_worked == shift:
                            weight = 0.0
                        else:
                            weight = 1.0
                    
                    if shift != 0 and weight > 0:
                        # Shift change
                        if rec_mode == "state_dependent":
                            h_val = (new_nu**gamma_C - (new_nu - 1)**gamma_C) if new_nu >= 1 else 0.0
                            delta_k = weight * h_val * K_max
                            new_k = min(e_max_int, k + delta_k)
                        else:
                            chg_amt = weight * K_max
                            if chg_mode == "convex":
                                chg_amt = chg_amt * (1.0 + 0.5 * k)
                            elif chg_mode == "concave":
                                chg_amt = chg_amt * (1.0 / (1.0 + 0.5 * k))
                            new_k = min(K_max, k + chg_amt)
                    else:
                        # No shift change
                        if rec_mode == "state_dependent":
                            r_val = 0.0
                            if new_h >= chi + 1:
                                r_val = alpha_R * ((new_h - chi)**gamma_R - (new_h - chi - 1)**gamma_R)
                            new_k = max(0, k - r_val * K_max)
                        else:
                            r_new = 0.0
                            if new_h >= chi + 1:
                                r_new = 1.0 * K_max
                            if rec_mode == "slow":
                                r_new = (1.0 * K_max) if new_h >= chi + 2 else 0.0
                            new_k = max(0, k - r_new)
                    
                    new_k = max(0, min(int(round(new_k)), K_max))
                    GammaF[k, h, nu, tau] = new_k
                    
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
        
    return GammaF, GammaH, GammaNu, pi_k
