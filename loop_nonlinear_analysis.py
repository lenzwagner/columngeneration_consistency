import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import column_generation_behavior
from Utils.gcutil import generate_dict_from_excel
from nonlinear_config import experiment_config, nonlinear_specs

def run_model(family, model_name, data, scaled_demand_dict, solver_config, I, T, K, prob, nl_spec=None):
    eps = 0.06
    chi = 3
    
    t0 = time.time()
    
    try:
        # Call the column generation behavior with an additional nl_spec parameter
        res = column_generation_behavior(
            data, scaled_demand_dict, eps, Min_WD_i, Max_WD_i, 
            solver_config["time_limit_init"] if "time_limit_init" in solver_config else 5, 
            200, 100, chi, solver_config["rc_tol"], solver_config["time_limit"], 
            I, T, K, prob, sp_solver='labeling_bidir', use_null_column=True,
            enforce_no_change=False, enforce_performance_floor=None,
            nl_spec=nl_spec # We will add this parameter to cg_behavior and downstream
        )
        time_total = time.time() - t0
        
        # Unpack results (based on the huge tuple returned by cg_behavior)
        # 0: undercoverage, 1: understaffing, 2: perfloss, 3: consistency
        uc = res[0]
        uc_perf = res[2]
        sc_total = res[3]
        
        final_obj = res[8]
        final_lb = res[9]
        cg_iterations = res[10]
        gap = res[12]
        time_sp = res[13]
        time_rmp = res[14]
        time_ip = res[15]
        ls_p = res[16]
        
        # Depletion metrics
        n_days = len(T)
        n_workers = len(I)
        end_day_indices = [(i * n_days) + (n_days - 1) for i in range(n_workers)]
        p_end_vals = [ls_p[idx] for idx in end_day_indices if idx < len(ls_p)]
        
        P_end = sum(p_end_vals) / len(p_end_vals) if p_end_vals else 0
        tau = 0.8
        B_end = sum(1 for p in p_end_vals if p < tau) / len(p_end_vals) if p_end_vals else 0
        
        k_val = 7
        tail_duration = 0
        for d in range(n_days - k_val + 1, n_days + 1):
            day_idx = d - 1
            p_day_vals = [ls_p[i * n_days + day_idx] for i in range(n_workers) if (i * n_days + day_idx) < len(ls_p)]
            P_d = sum(p_day_vals) / len(p_day_vals) if p_day_vals else 1.0
            if P_d < tau:
                tail_duration += 1
                
        return {
            "status": "Optimal" if gap < solver_config["mip_gap"] * 100 else "Suboptimal",
            "runtime_total": time_total,
            "runtime_mp": time_rmp,
            "runtime_sp": time_sp,
            "runtime_ip": time_ip,
            "cg_iterations": cg_iterations,
            "final_gap": gap,
            "objective": final_obj,
            "U": uc,
            "U_inherent": uc - uc_perf, # Approximation, U_inherent + U_perf = U
            "U_perf": uc_perf,
            "SC_total": sc_total,
            "P_end": P_end,
            "E_end": B_end, # Can map to same or different if needed
            "B_end": B_end,
            "L_tail_k7": tail_duration,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Model failed: {e}")
        return {
            "status": "Error/Infeasible",
            "runtime_total": time.time() - t0,
            "runtime_mp": np.nan,
            "runtime_sp": np.nan,
            "runtime_ip": np.nan,
            "cg_iterations": np.nan,
            "final_gap": np.nan,
            "objective": np.nan,
            "U": np.nan,
            "U_inherent": np.nan,
            "U_perf": np.nan,
            "SC_total": np.nan,
            "P_end": np.nan,
            "E_end": np.nan,
            "B_end": np.nan,
            "L_tail_k7": np.nan,
        }

def compare_runs(baseline, nl_res):
    deltas = {}
    for key in ["U", "U_inherent", "U_perf", "SC_total", "P_end", "E_end", "B_end", "L_tail_k7", "runtime_total", "final_gap"]:
        if pd.notna(baseline.get(key)) and pd.notna(nl_res.get(key)):
            deltas[f"delta_{key}"] = nl_res[key] - baseline[key]
        else:
            deltas[f"delta_{key}"] = np.nan
    return deltas

def run_nonlinear_analysis():
    print("="*80)
    print("NON-LINEAR BEHAVIORAL ANALYSIS")
    print("="*80)
    
    solver_config = experiment_config["solver"]
    
    # We set up general data based on m5
    # m5 implies n_workers = 100, n_days = 28 usually in this project
    n_days = 28
    n_workers = 100
    T = list(range(1, n_days + 1))
    I = list(range(1, n_workers + 1))
    K = [1, 2, 3] # Shift types
    
    all_results = []
    
    for instance in experiment_config["instances"]:
        # Extract number of workers from instance string (e.g. "m5" -> 5)
        if instance.startswith("m"):
            n_workers = int(instance[1:])
        else:
            n_workers = 100
            
        I = list(range(1, n_workers + 1))
        
        data = pd.DataFrame({
            'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
            'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
            'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
        })
        
        for scenario_id in experiment_config["scenario_ids"]:
            
            try:
                base_demand_dict = generate_dict_from_excel(
                    'data/demand_data.xlsx', n_workers, 'Medium', scenario=scenario_id + 1
                )
                base_demand_dict = {k: v for k, v in base_demand_dict.items() if k[0] <= n_days}
            except Exception as e:
                print(f"Failed to pull base demand for seed {scenario_id}: {e}")
                continue
                
            for lam in experiment_config["lambdas"]:
                print(f"\n[{instance} | Scenario {scenario_id} | Lambda {lam:.2f}]")
                scaled_demand_dict = {k: int(round(v * lam)) for k, v in base_demand_dict.items()}
                
                # Run Baseline
                print(f"  -> Running LINEAR_BASE")
                baseline_result = run_model(
                    family="linear",
                    model_name="LINEAR_BASE",
                    data=data,
                    scaled_demand_dict=scaled_demand_dict,
                    solver_config=solver_config,
                    I=I, T=T, K=K, prob=1.0,
                    nl_spec=None
                )
                
                base_row = {
                    "instance": instance,
                    "scenario_id": scenario_id,
                    "lambda": lam,
                    "model_family": "linear",
                    "model_name": "LINEAR_BASE",
                    **baseline_result
                }
                all_results.append(base_row)
                
                # Run Non-Linear Specs
                for spec in experiment_config["model_specs"]:
                    if spec["family"] == "nonlinear":
                        print(f"  -> Running {spec['name']}")
                        nl_spec_data = nonlinear_specs[spec["name"]]
                        
                        nl_result = run_model(
                            family="nonlinear",
                            model_name=spec["name"],
                            data=data,
                            scaled_demand_dict=scaled_demand_dict,
                            solver_config=solver_config,
                            I=I, T=T, K=K, prob=1.0,
                            nl_spec=nl_spec_data
                        )
                        
                        deltas = compare_runs(baseline_result, nl_result)
                        
                        nl_row = {
                            "instance": instance,
                            "scenario_id": scenario_id,
                            "lambda": lam,
                            "model_family": "nonlinear",
                            "model_name": spec["name"],
                            **nl_result,
                            **deltas
                        }
                        all_results.append(nl_row)

    df = pd.DataFrame(all_results)
    os.makedirs('results/nonlinear', exist_ok=True)
    out_path = f'results/nonlinear/nonlinear_analysis_{datetime.now().strftime("%d_%m_%Y_%H-%M")}.csv'
    df.to_csv(out_path, index=False)
    print(f"\nAnalysis complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_nonlinear_analysis()
