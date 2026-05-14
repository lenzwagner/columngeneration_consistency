import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import column_generation_behavior
from Utils.gcutil import generate_dict_from_excel
from nonlinear_evaluation import evaluate_worker_schedule, evaluate_nonlinear_schedule

# 1. Configuration
n_workers = 5
n_days = 28
T = list(range(1, n_days + 1))
I = list(range(1, n_workers + 1))
K = [1, 2, 3] # Early, Late, Night

scenarios = list(range(25)) # 0 to 24
eps_baseline = 0.06
chi = 3

solver_config = {
    "time_limit": 7200,
    "mip_gap": 0.01,
    "threads": 1,
    "seed": 42,
    "rc_tol": 1e-5,
    "time_limit_init": 5
}

delta_matrices = {
    "D0": {
        # Symmetric baseline
        1: {1: 0.00, 2: 0.06, 3: 0.06},
        2: {1: 0.06, 2: 0.00, 3: 0.06},
        3: {1: 0.06, 2: 0.06, 3: 0.00}
    },
    "D1": {
        # Mild asymmetric
        1: {1: 0.00, 2: 0.06, 3: 0.08},
        2: {1: 0.08, 2: 0.00, 3: 0.05},
        3: {1: 0.12, 2: 0.08, 3: 0.00}
    },
    "D2": {
        # Strong asymmetric
        1: {1: 0.00, 2: 0.06, 3: 0.13},
        2: {1: 0.11, 2: 0.00, 3: 0.08},
        3: {1: 0.20, 2: 0.10, 3: 0.00}
    }
}

betas = {"fast": 0.3, "medium": 0.6, "slow": 0.85}

def run_study():
    os.makedirs('results/nonlinear', exist_ok=True)
    
    run_summaries = []
    all_schedules = []
    all_perf_states = []
    all_coverage = []
    all_rotation_metrics = []
    parameter_grid = []
    
    data = pd.DataFrame({
        'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
        'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
        'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
    })
    
    configs_to_run = [
        # (d_id, config_name, d_mat, alpha_R, gamma_R, gamma_C, e_max)
        # Main Analysis (Concave recovery, Convex change)
        ("D2", "base_nonlinear", delta_matrices["D2"], 0.04, 0.5, 1.25, 0.50),
        
        # Form-Kombinationen (Shape Variations)
        ("D2", "pure_concave", delta_matrices["D2"], 0.04, 0.5, 0.8, 0.50),   
        ("D2", "pure_convex", delta_matrices["D2"], 0.04, 1.25, 1.25, 0.50),  
        ("D2", "pure_linear", delta_matrices["D2"], 0.04, 1.0, 1.0, 0.50),    
        ("D2", "convex_rec_concave_chg", delta_matrices["D2"], 0.04, 1.25, 0.8, 0.50), 
        ("D2", "concave_rec_convex_chg", delta_matrices["D2"], 0.04, 0.5, 1.25, 0.50), 
        
        # Intensitäts-Variationen (alpha_R)
        ("D2", "slow_recovery", delta_matrices["D2"], 0.02, 0.5, 1.25, 0.50),
        ("D2", "fast_recovery", delta_matrices["D2"], 0.08, 0.5, 1.25, 0.50),
        ("D2", "v_fast_recovery", delta_matrices["D2"], 0.12, 0.5, 1.25, 0.50),
        
        # Krümmungs-Variationen (gamma_R)
        ("D2", "extreme_concave_rec", delta_matrices["D2"], 0.04, 0.2, 1.25, 0.50),
        ("D2", "strong_convex_rec", delta_matrices["D2"], 0.04, 1.5, 1.25, 0.50),
        
        # Krümmungs-Variationen (gamma_C)
        ("D2", "extreme_concave_chg", delta_matrices["D2"], 0.04, 0.5, 0.5, 0.50),
        ("D2", "strong_convex_chg", delta_matrices["D2"], 0.04, 0.5, 1.75, 0.50),
        ("D2", "extreme_convex_chg", delta_matrices["D2"], 0.04, 0.5, 2.25, 0.50),

        # Kapazitäts-Variationen (e_max)
        ("D2", "low_fatigue_cap", delta_matrices["D2"], 0.04, 0.5, 1.25, 0.30),
        ("D2", "high_fatigue_cap", delta_matrices["D2"], 0.04, 0.5, 1.25, 0.80),

        # Matrix/Asymmetrie Variationen
        ("D0", "symmetric", delta_matrices["D0"], 0.04, 0.5, 1.25, 0.50),
        ("D1", "mild_asymmetric", delta_matrices["D1"], 0.04, 0.5, 1.25, 0.50),
    ]
    
    for d_id, b_id, d_mat, alpha_R, gamma_R, gamma_C, e_max in configs_to_run:
        parameter_grid.append({
            "delta_matrix_id": d_id,
            "alpha_R": alpha_R,
            "gamma_R": gamma_R,
            "gamma_C": gamma_C,
            "e_max": e_max,
            "rho_max": "inf",
            "grid_step": 0.01
        })
        for s1 in [1, 2, 3]:
            for s2 in [1, 2, 3]:
                if s1 != s2:
                    parameter_grid.append({
                        "delta_matrix_id": d_id,
                        "from_shift": s1,
                        "to_shift": s2,
                        "delta": d_mat[s1][s2]
                    })
    
    pd.DataFrame(parameter_grid).to_csv('results/nonlinear/parameter_grid.csv', index=False)
    
    for scenario_id in scenarios: 
        print(f"Scenario {scenario_id}")
        try:
            base_demand_dict = generate_dict_from_excel(
                'data/demand_data.xlsx', 50, 'Medium', scenario=scenario_id + 1
            )
            # Scale from 50 to 5 workers
            base_demand_dict = {k: max(1, int(round(v / 10.0))) for k, v in base_demand_dict.items() if k[0] <= n_days}
        except Exception as e:
            print(f"Failed to pull base demand for seed {scenario_id}: {e}")
            continue
            
        for d_id, b_id, d_mat, alpha_R, gamma_R, gamma_C, e_max in configs_to_run:
            nl_spec = {
                "name": "NL_STATE_DEP",
                "fatigue_levels": 100,
                "stability_levels": 28,
                "chg_mode": "matrix",
                "rec_mode": "state_dependent",
                "perf_mode": "linear",
                "alpha_R": alpha_R,
                "gamma_R": gamma_R,
                "gamma_C": gamma_C,
                "e_max": e_max,
                "delta_matrix": d_mat
            }
            
            # Variant A: NPP
            print(f"  [{d_id}-{b_id}] Variant A: NPP")
            nl_spec_npp = {
                "name": "NL_STATE_DEP",
                "fatigue_levels": 100,
                "stability_levels": 28,
                "chg_mode": "matrix",
                "rec_mode": "state_dependent",
                "perf_mode": "linear",
                "alpha_R": 0.0,
                "gamma_R": 1.0,
                "gamma_C": 1.0,
                "e_max": 0.0,
                "delta_matrix": {1: {1:0,2:0,3:0}, 2: {1:0,2:0,3:0}, 3: {1:0,2:0,3:0}}
            }
            t0 = time.time()
            res_a = column_generation_behavior(
                data, base_demand_dict, eps_baseline, Min_WD_i, Max_WD_i, 
                solver_config["time_limit_init"], 200, 100, chi, solver_config["rc_tol"], solver_config["time_limit"], 
                I, T, K, 1.0, sp_solver='labeling_bidir', use_null_column=True,
                nl_spec=nl_spec_npp
            )
            rt_a = time.time() - t0
            ls_x_a = res_a[19] # schedule ls_x flat list
            x_a = {}
            idx = 0
            for i in I:
                for d in T:
                    for k in K:
                        x_a[(i, d, k)] = ls_x_a[idx]
                        idx += 1
            states_a = evaluate_nonlinear_schedule(x_a, nl_spec, T, I, K)
            
            # Variant B: Linear BAP
            print(f"  [{d_id}-{b_id}] Variant B: Linear BAP")
            t0 = time.time()
            res_b = column_generation_behavior(
                data, base_demand_dict, eps_baseline, Min_WD_i, Max_WD_i, 
                solver_config["time_limit_init"], 200, 100, chi, solver_config["rc_tol"], solver_config["time_limit"], 
                I, T, K, 1.0, sp_solver='labeling_bidir', use_null_column=True
            )
            rt_b = time.time() - t0
            ls_x_b = res_b[19]
            x_b = {}
            idx = 0
            for i in I:
                for d in T:
                    for k in K:
                        x_b[(i, d, k)] = ls_x_b[idx]
                        idx += 1
            states_b = evaluate_nonlinear_schedule(x_b, nl_spec, T, I, K)
            
            # Variant C: Nonlinear BAP
            print(f"  [{d_id}-{b_id}] Variant C: Nonlinear BAP")
            t0 = time.time()
            res_c = column_generation_behavior(
                data, base_demand_dict, eps_baseline, Min_WD_i, Max_WD_i, 
                solver_config["time_limit_init"], 200, 100, chi, solver_config["rc_tol"], solver_config["time_limit"], 
                I, T, K, 1.0, sp_solver='labeling_bidir', use_null_column=True,
                nl_spec=nl_spec
            )
            rt_c = time.time() - t0
            ls_x_c = res_c[19]
            x_c = {}
            idx = 0
            for i in I:
                for d in T:
                    for k in K:
                        x_c[(i, d, k)] = ls_x_c[idx]
                        idx += 1
            states_c = evaluate_nonlinear_schedule(x_c, nl_spec, T, I, K)
            
            # Helper to compile metrics
            def compile_metrics(var_name, x_dict, states_df, rt, cg_res):
                # schedules
                for i in I:
                    for d in T:
                        for k in K:
                            val = x_dict.get((i, d, k), 0)
                            if val > 0.5:
                                all_schedules.append({
                                    "scenario_id": scenario_id,
                                    "model_variant": var_name,
                                    "worker_id": i,
                                    "day": d,
                                    "shift": k,
                                    "x": 1
                                })
                
                # states
                states_df['scenario_id'] = scenario_id
                states_df['model_variant'] = var_name
                all_perf_states.append(states_df)
                
                # coverage
                u_tot = 0
                u_inh = 0
                u_per = 0
                for d in T:
                    for k in K:
                        dem = base_demand_dict.get((d, k), 0)
                        nom_sup = sum(1 for i in I if x_dict.get((i, d, k), 0) > 0.5)
                        # effective supply
                        eff_sup = states_df[(states_df['day'] == d) & (states_df['shift'] == k)]['performance'].sum()
                        
                        inh = max(0, dem - nom_sup)
                        tot = max(0, dem - eff_sup)
                        per = tot - inh
                        
                        u_tot += tot
                        u_inh += inh
                        u_per += per
                        
                        all_coverage.append({
                            "scenario_id": scenario_id,
                            "model_variant": var_name,
                            "day": d,
                            "shift": k,
                            "demand": dem,
                            "nominal_supply": nom_sup,
                            "effective_supply": eff_sup,
                            "inherent_undercoverage": inh,
                            "performance_undercoverage": per,
                            "total_undercoverage": tot
                        })
                
                # rotation
                shift_changes = states_df['shift_change'].sum()
                backward_rot = len(states_df[
                    ((states_df['previous_shift'] == 2) & (states_df['shift'] == 1)) |
                    ((states_df['previous_shift'] == 3) & (states_df['shift'] == 1)) |
                    ((states_df['previous_shift'] == 3) & (states_df['shift'] == 2))
                ])
                n_to_e = len(states_df[(states_df['previous_shift'] == 3) & (states_df['shift'] == 1)])
                fwd_rot = shift_changes - backward_rot
                
                avg_stable = states_df[states_df['shift_change'] == 1]['rho'].mean() if shift_changes > 0 else 0
                mean_perf_loss = states_df[states_df['shift'] > 0]['e_after'].mean()
                max_perf_loss = states_df['e_after'].max()
                
                for i in I:
                    wdf = states_df[states_df['worker_id'] == i]
                    sc = wdf['shift_change'].sum()
                    br = len(wdf[
                        ((wdf['previous_shift'] == 2) & (wdf['shift'] == 1)) |
                        ((wdf['previous_shift'] == 3) & (wdf['shift'] == 1)) |
                        ((wdf['previous_shift'] == 3) & (wdf['shift'] == 2))
                    ])
                    all_rotation_metrics.append({
                        "scenario_id": scenario_id,
                        "model_variant": var_name,
                        "worker_id": i,
                        "total_shift_changes": sc,
                        "forward_rotations": sc - br,
                        "backward_rotations": br,
                        "night_to_early": len(wdf[(wdf['previous_shift'] == 3) & (wdf['shift'] == 1)]),
                        "early_to_night": len(wdf[(wdf['previous_shift'] == 1) & (wdf['shift'] == 3)]),
                        "avg_stable_block_length": wdf[wdf['shift_change'] == 1]['rho'].mean() if sc > 0 else 0
                    })
                
                run_summaries.append({
                    "scenario_id": scenario_id,
                    "model_variant": var_name,
                    "delta_matrix_id": d_id,
                    "alpha_R": alpha_R,
                    "gamma_R": gamma_R,
                    "gamma_C": gamma_C,
                    "e_max": 1.0,
                    "objective_value": cg_res[8],
                    "total_undercoverage": u_tot,
                    "inherent_undercoverage": u_inh,
                    "performance_undercoverage": u_per,
                    "total_shift_changes": shift_changes,
                    "backward_rotations": backward_rot,
                    "night_to_early": n_to_e,
                    "avg_stable_block_length": avg_stable,
                    "mean_performance_loss": mean_perf_loss,
                    "max_worker_performance_loss": max_perf_loss,
                    "gini_performance_loss": 0, # Placeholder
                    "runtime_total": rt,
                    "runtime_mp": cg_res[14],
                    "runtime_sp": cg_res[13],
                    "runtime_ip": cg_res[15],
                    "cg_iterations": cg_res[10],
                    "final_rmp_gap": cg_res[12]
                })

            compile_metrics("NPP", x_a, states_a, rt_a, res_a)
            compile_metrics("Linear BAP", x_b, states_b, rt_b, res_b)
            compile_metrics("Nonlinear BAP", x_c, states_c, rt_c, res_c)

    # Save outputs
    pd.DataFrame(run_summaries).to_csv('results/nonlinear/run_summary.csv', index=False)
    pd.DataFrame(all_schedules).to_csv('results/nonlinear/schedules.csv', index=False)
    pd.concat(all_perf_states).to_csv('results/nonlinear/performance_states.csv', index=False)
    pd.DataFrame(all_coverage).to_csv('results/nonlinear/coverage_by_shift.csv', index=False)
    pd.DataFrame(all_rotation_metrics).to_csv('results/nonlinear/rotation_metrics.csv', index=False)

if __name__ == '__main__':
    run_study()
