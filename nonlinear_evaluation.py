import pandas as pd
import numpy as np

def R_cumulative(rho, alpha, beta, chi):
    """Cumulative recovery after rho stable days."""
    if rho <= chi:
        return 0.0
    # The prompt states R(rho) = alpha(1 - beta^rho).
    # But usually recovery starts after chi days.
    # So it should be R(rho) = alpha(1 - beta^(rho - chi))
    # Let's use (rho - chi) to make the exponent start at 1 when rho = chi + 1
    return alpha * (1.0 - beta ** (rho - chi))

def r_daily(rho, alpha, beta, chi):
    """Daily recovery increment."""
    if rho <= chi:
        return 0.0
    return R_cumulative(rho, alpha, beta, chi) - R_cumulative(rho - 1, alpha, beta, chi)

def evaluate_worker_schedule(schedule, delta_matrix, alpha, beta, chi=3, e_max=1.0):
    """
    Evaluate a single worker's schedule.
    schedule: list of shifts [None, 1, 1, 2, ...] where None or 0 means Off.
    Returns detailed states per day.
    """
    states = []
    
    e = 0.0
    rho = 0
    last_worked_shift = None
    
    for day, shift in enumerate(schedule, 1):
        if shift == 0:
            shift = None
            
        e_before = e
        shift_change = False
        delta = 0.0
        recovery = 0.0
        transition_type = "None"
        
        if shift is not None:
            # Worked a shift
            if last_worked_shift is not None and last_worked_shift != shift:
                # Shift change!
                shift_change = True
                rho = 0
                delta = delta_matrix.get(last_worked_shift, {}).get(shift, 0.0)
                e = min(e_max, e + delta)
                transition_type = "Change"
            else:
                # No shift change
                rho += 1
                recovery = r_daily(rho, alpha, beta, chi)
                e = max(0.0, e - recovery)
                transition_type = "Same"
                
            last_worked_shift = shift
            performance = 1.0 - e
        else:
            # Day off
            rho += 1
            recovery = r_daily(rho, alpha, beta, chi)
            e = max(0.0, e - recovery)
            transition_type = "Off"
            performance = 0.0 # x=0
            
        states.append({
            'day': day,
            'shift': shift if shift is not None else 0,
            'previous_shift': last_worked_shift if transition_type == "Change" else (last_worked_shift if shift is not None else 0),
            'rho': rho,
            'e_before': e_before,
            'delta': delta,
            'recovery': recovery,
            'e_after': e,
            'performance': performance,
            'shift_change': 1 if shift_change else 0,
            'transition_type': transition_type
        })
        
    return states

def evaluate_nonlinear_schedule(x_dict, nl_spec, T, I, K):
    """
    Evaluate a full schedule (x_dict).
    x_dict: {(day, shift): [binary values for each worker, or dict keyed by worker]}
    For our output format, x_dict is usually {(day, shift): [w1, w2, ...]}
    But we need it by worker.
    """
    # Assuming x_dict is {(day, shift): sum(x)} or {(i, day, shift): value}
    # We will accept {(i, day, shift): 1 or 0}
    
    delta_matrix = nl_spec.get('delta_matrix', {})
    alpha = nl_spec.get('alpha', 1.0)
    beta = nl_spec.get('beta', 1.0)
    chi = nl_spec.get('stability_levels', 14) # Default to something, or 3
    # usually chi is 3 in this setup
    chi = 3
    e_max = 1.0
    
    all_states = []
    
    for i in I:
        # Reconstruct worker schedule
        schedule = [0] * len(T)
        for t_idx, day in enumerate(T):
            worked_shift = 0
            for shift in K:
                if x_dict.get((i, day, shift), 0) > 0.5:
                    worked_shift = shift
                    break
            schedule[t_idx] = worked_shift
            
        states = evaluate_worker_schedule(schedule, delta_matrix, alpha, beta, chi, e_max)
        for s in states:
            s['worker_id'] = i
            all_states.append(s)
            
    return pd.DataFrame(all_states)

def check_1_handtest():
    """
    Test E E E E L L L N E
    E=1, L=2, N=3
    """
    print("Running Check 1: Handtest E E E E L L L N E")
    schedule = [1, 1, 1, 1, 2, 2, 2, 3, 1]
    
    delta_matrix = {
        1: {1: 0.00, 2: 0.06, 3: 0.13},
        2: {1: 0.11, 2: 0.00, 3: 0.08},
        3: {1: 0.20, 2: 0.10, 3: 0.00}
    }
    
    alpha = 0.3
    beta = 0.8
    chi = 3
    
    states = evaluate_worker_schedule(schedule, delta_matrix, alpha, beta, chi)
    df = pd.DataFrame(states)
    print(df[['day', 'shift', 'transition_type', 'rho', 'delta', 'recovery', 'e_after']])
    
if __name__ == "__main__":
    check_1_handtest()
