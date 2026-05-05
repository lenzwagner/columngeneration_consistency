import numpy as np

def classify_transition(last_worked, shift):
    """
    Classify the transition between last_worked shift and current shift.
    0: off (not handled here, handled separately or as tau=0)
    1: same
    2: mild (e.g. morning to evening)
    3: severe (e.g. evening to morning, or any backward rotation)
    4: return (off to work)
    """
    if last_worked == 0:
        return 4 # return
    if last_worked == shift:
        return 1 # same
    
    # Simple classification for mild vs severe
    # Assuming shifts: 1 (Morning), 2 (Evening), 3 (Night)
    if last_worked < shift:
        return 2 # mild (forward rotation)
    else:
        return 3 # severe (backward rotation)

def generate_transitions(eps, chi, omega_max, nl_spec=None):
    """
    Precompute the GammaF, GammaH matrices and pi_k array.
    If nl_spec is None, it generates the linear baseline equivalent.
    """
    K_max = omega_max
    H_max = chi + 5 # sufficient upper bound for stability tracking
    
    # tau: 0=off, 1=same, 2=mild, 3=severe, 4=return
    n_tau = 5
    
    GammaF = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
    GammaH = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
    pi_k = np.zeros(K_max + 1, dtype=np.float64)
    
    # Transition weights default (linear)
    w_same = 0.0
    w_mild = 1.0
    w_severe = 1.0
    w_return = 1.0 # originally, after day off c_new=0 or 1? Wait, in linear: last_worked=0, shift>0 -> c_new=1 if last_worked > 0. So c_new=0 when return!
    
    chg_mode = "linear"
    rec_mode = "linear"
    perf_mode = "linear"
    
    if nl_spec is not None:
        K_max = nl_spec.get("fatigue_levels", K_max)
        H_max = nl_spec.get("stability_levels", H_max)
        GammaF = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
        GammaH = np.zeros((K_max + 1, H_max + 1, n_tau), dtype=np.int32)
        pi_k = np.zeros(K_max + 1, dtype=np.float64)
        
        tw = nl_spec.get("transition_weights", {})
        w_same = tw.get("same", 0.0)
        w_mild = tw.get("mild", 1.0)
        w_severe = tw.get("severe", 1.0)
        w_return = tw.get("return", 0.0)
        
        chg_mode = nl_spec.get("chg_mode", "linear")
        rec_mode = nl_spec.get("rec_mode", "linear")
        perf_mode = nl_spec.get("perf_mode", "linear")
    else:
        # Linear Baseline exactly matches the old inline logic:
        # c_new = 1 if last_worked > 0 and last_worked != shift else 0
        w_same = 0.0
        w_mild = 1.0
        w_severe = 1.0
        w_return = 0.0
        
    for k in range(K_max + 1):
        for h in range(H_max + 1):
            for tau in range(n_tau):
                
                # 1. Determine base change weight
                if tau == 0: # day off
                    weight = 0.0
                elif tau == 1:
                    weight = w_same
                elif tau == 2:
                    weight = w_mild
                elif tau == 3:
                    weight = w_severe
                elif tau == 4:
                    weight = w_return
                    
                # 2. Update H (stability)
                new_h = h + 1
                if tau != 0: # working
                    if weight > 0: # disruption resets stability
                        new_h = 0
                else: # day off, we don't reset stability, we just increment rho? Wait.
                    pass # rho increments on day off in linear
                    
                new_h = min(new_h, H_max)
                GammaH[k, h, tau] = new_h
                
                # 3. Determine Recovery
                r_new = 0
                if new_h >= chi + 1:
                    r_new = 1
                    
                if rec_mode == "slow":
                    # e.g., only recover if h >= chi + 2
                    r_new = 1 if new_h >= chi + 2 else 0
                
                # 4. Determine new Fatigue
                chg_amt = weight
                if chg_mode == "convex":
                    # disruption hurts more at higher fatigue
                    chg_amt = weight * (1.0 + 0.5 * k)
                elif chg_mode == "concave":
                    # disruption hurts less at higher fatigue
                    chg_amt = weight * (1.0 / (1.0 + 0.5 * k))
                
                new_k = k + chg_amt - r_new
                
                # Bounds
                new_k = max(0, min(int(round(new_k)), K_max))
                GammaF[k, h, tau] = new_k
                
        # Performance
        xi = 1 - eps * omega_max
        if perf_mode == "linear":
            kappa = 1 if k >= omega_max else 0
            p = 1.0 - eps * k - xi * kappa
        elif perf_mode == "threshold":
            # sudden drop if k > K_max / 2
            if k > K_max / 2:
                p = 1.0 - eps * k - 0.2 # extra penalty
            else:
                p = 1.0 - eps * k
        else:
            p = 1.0 - eps * k
            
        p = max(0.0, min(p, 1.0))
        pi_k[k] = p
        
    return GammaF, GammaH, pi_k
