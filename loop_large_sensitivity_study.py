import pandas as pd
import numpy as np
import time
import os
import csv
from cg_behavior import column_generation_behavior
from worker_groups import create_homogeneous_group
from nonlinear_transitions import evaluate_schedule_nl, h_func, r_func
from Utils.gcutil import generate_dict_from_excel

def get_matrices(epsilon):
    """Defines the three delta matrices."""
    base = np.zeros((4, 4))
    base[1,1]=1.0; base[1,2]=1.2; base[1,3]=1.5
    base[2,1]=2.5; base[2,2]=1.0; base[2,3]=1.2
    base[3,1]=1.2; base[3,2]=1.5; base[3,3]=1.0
    
    severe = np.zeros((4, 4))
    severe[1,1]=1.0; severe[1,2]=2.0; severe[1,3]=2.5
    severe[2,1]=5.0; severe[2,2]=1.0; severe[2,3]=2.0
    severe[3,1]=2.0; severe[3,2]=2.5; severe[3,3]=1.0
    
    close = np.zeros((4, 4))
    close[1,1]=1.0; close[1,2]=1.1; close[1,3]=1.2
    close[2,1]=1.5; close[2,2]=1.0; close[2,3]=1.1
    close[3,1]=1.1; close[3,2]=1.2; close[3,3]=1.0
    
    # Apply epsilon scaling
    for m in [base, severe, close]:
        m *= epsilon
        
    return {
        'base': base,
        'severe': severe,
        'close': close
    }

def run_experiment():
    I_sizes = [100]
    scarcities = [('Medium', 1.0)]
    seeds = range(1, 2)
    
    # Parameter Sets
    epsilon = 0.06
    matrices_dict = get_matrices(epsilon)
    chis = [1, 3, 5, 7]
    alphas = [0.02, 0.04, 0.08]
    
    steps_convex = [1.2, 1.5, 2.0]
    steps_concave = [0.5, 0.7, 0.9]
    
    types = {
        'conv': (steps_convex, steps_convex),
        'conc': (steps_concave, steps_concave),
        'convconc': (steps_concave, steps_convex),
        'concconv': (steps_convex, steps_concave)
    }
    
    csv_file = 'sensitivity_results_100_med.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['seed', 'matrix', 'chi', 'alpha', 'type', 'gamma_r', 'gamma_c', 
                             'bap_obj', 'npp_nom', 'npp_expost', 'improvement_pct', 'time'])
        
        total_configs = len(matrices_dict) * len(chis) * len(alphas) * 4 * 9
        total_runs = total_configs * len(seeds)
        count = 0
        
        for seed in seeds:
            # LOAD DEMAND FROM EXCEL (AS IN OTHER SCRIPTS)
            demand = generate_dict_from_excel('data/demand_data.xlsx', 100, 'Medium', scenario=seed)
            
            if not demand:
                print(f"Skipping Seed {seed} - No demand data found.")
                continue

            T_days = 28
            T = list(range(1, T_days + 1))
            K = [1, 2, 3]
            I = list(range(1, 101))
            
            df_placeholder = pd.DataFrame([{'T': t, 'K': k, 'I': 1} for t in T for k in K])

            for m_name, delta in matrices_dict.items():
                for chi in chis:
                    for alpha in alphas:
                        for t_name, (r_steps, c_steps) in types.items():
                            for gr in r_steps:
                                for gc in c_steps:
                                    count += 1
                                    if count % 10 == 0 or count == 1:
                                        print(f"Run {count}/{total_runs} (Seed {seed}): {m_name}, chi={chi}, {t_name}")
                                    
                                    nl_spec = {
                                        'epsilon': epsilon,
                                        'chi': chi,
                                        'gamma_R': gr,
                                        'gamma_C': gc,
                                        'alpha_R': alpha,
                                        'delta': delta,
                                        'e_max': 0.5
                                    }
                                    
                                    start_t = time.time()
                                    
                                    # 1. BAP
                                    bap_res = column_generation_behavior(
                                        df_placeholder, demand, epsilon, 4, 6, 10, 100, 0, chi, 0.001, 10, I, T, K, 1.0,
                                        sp_solver="labeling", nl_spec=nl_spec
                                    )
                                    bap_obj = bap_res[8]
                                    
                                    # 2. NPP
                                    npp_res = column_generation_behavior(
                                        df_placeholder, demand, 0.0, 4, 6, 10, 100, 0, chi, 0.001, 10, I, T, K, 1.0,
                                        sp_solver="labeling", nl_spec=None
                                    )
                                    npp_nom = npp_res[8]
                                    
                                    # 3. NPP Ex-Post
                                    ls_x = npp_res[19]
                                    real_supply = [0.0] * (len(T) * 3)
                                    for idx in range(100):
                                        worker_x = ls_x[idx * (len(T)*3) : (idx+1) * (len(T)*3)]
                                        x_dict = {(T[d], K[s]): 1.0 for d in range(len(T)) for s in range(3) if worker_x[d*3+s] > 0.5}
                                        p_hist, _, _, _ = evaluate_schedule_nl(x_dict, T, K, nl_spec)
                                        for d in range(len(T)):
                                            p = p_hist[T[d]]
                                            for s in range(3):
                                                if worker_x[d*3+s] > 0.5:
                                                    real_supply[d*3+s] += p
                                    
                                    demand_vals = [demand.get((t, k), 0) for t in T for k in K]
                                    npp_expost = sum(max(0, demand_vals[i] - real_supply[i]) for i in range(len(demand_vals)))
                                    
                                    imp = ((npp_expost - bap_obj) / npp_expost * 100) if npp_expost > 0 else 0
                                    elapsed = time.time() - start_t
                                    
                                    writer.writerow([seed, m_name, chi, alpha, t_name, gr, gc, bap_obj, npp_nom, npp_expost, imp, elapsed])
                                    f.flush()

if __name__ == "__main__":
    run_experiment()
