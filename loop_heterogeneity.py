"""
Heterogeneity Experiment Loop

Runs 5 experiment configurations across 25 seeds each:
1. Basis-Lauf (H1 & H3): Mixed team (33+33+34) at Q=1.0
2. Referenz-Lauf (H3): Homogeneous team (100 G2) at Q=1.0
3. Stress-Szenario (H3): Mixed team at Q=1.1
4. Narrow Gap: ε ∈ {0.08, 0.06, 0.04}, χ=3
5. Recovery Focus: ε=0.06, χ ∈ {1, 3, 5}
"""

from Utils.setup import Min_WD_i, Max_WD_i
from cg_behavior import *
from cg_naive import column_generation_naive
from Utils.Plots.plots import *
from Utils.aggundercover import *
from datetime import *
from Utils.demand import *
from worker_groups import create_groups_from_fractions, create_homogeneous_group
import time
from scipy import stats
import numpy as np

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

EXPERIMENTS = {
    'basis': {
        'name': 'Basis-Lauf (H1 & H3)',
        'description': 'Mixed team at medium demand',
        'team': 'mixed',
        'fractions': "1/3,1/3,1/3",
        'group_params': [(0.10, 5), (0.06, 3), (0.02, 1)],  # G1, G2, G3
        'Q_scar': 1.0,
        'n_workers': 100,
    },
    'reference': {
        'name': 'Referenz-Lauf (H3)',
        'description': 'Homogeneous team (all G2) at medium demand',
        'team': 'homogeneous',
        'fractions': None,
        'group_params': [(0.06, 3)],  # All G2
        'Q_scar': 1.0,
        'n_workers': 100,
    },
    'stress': {
        'name': 'Stress-Szenario (H3)',
        'description': 'Mixed team at high demand',
        'team': 'mixed',
        'fractions': "1/3,1/3,1/3",
        'group_params': [(0.10, 5), (0.06, 3), (0.02, 1)],
        'Q_scar': 1.1,
        'n_workers': 100,
    },
    'narrow_gap': {
        'name': 'Narrow Gap Variation',
        'description': 'Reduced epsilon differences, fixed chi',
        'team': 'mixed',
        'fractions': "1/3,1/3,1/3",
        'group_params': [(0.08, 3), (0.06, 3), (0.04, 3)],  # Narrow ε gap
        'Q_scar': 1.0,
        'n_workers': 100,
    },
    'recovery_focus': {
        'name': 'Recovery Focus Variation',
        'description': 'Fixed epsilon, varied chi',
        'team': 'mixed',
        'fractions': "1/3,1/3,1/3",
        'group_params': [(0.06, 5), (0.06, 3), (0.06, 1)],  # Varied χ
        'Q_scar': 1.0,
        'n_workers': 100,
    },
}

# ============================================================================
# PARAMETERS
# ============================================================================

time_Limit, time_cg, time_cg_init = 7200, 7200, 60
max_itr, output_len, threshold = 2000, 98, 6e-5
N_SEEDS = 25  # Number of scenarios per experiment

# Which experiments to run (set to list of keys or 'all')
RUN_EXPERIMENTS = ['basis', 'reference']  # Start with these two for H3 comparison
# RUN_EXPERIMENTS = list(EXPERIMENTS.keys())  # Uncomment to run all

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_group_metrics(ls_sc, worker_groups, n_days):
    """Calculate shift changes per group."""
    group_metrics = {}
    for group_name, group in worker_groups.items():
        total_sc = 0
        for worker_id in group.worker_ids:
            w_idx = worker_id - 1
            sc_start = w_idx * n_days
            sc_end = sc_start + n_days
            if sc_end <= len(ls_sc):
                total_sc += sum(ls_sc[sc_start:sc_end])
        group_metrics[group_name] = {
            'total_shift_changes': total_sc,
            'avg_shift_changes': total_sc / len(group.worker_ids) if group.worker_ids else 0,
            'n_workers': len(group.worker_ids),
            'epsilon': group.epsilon,
            'chi': group.chi
        }
    return group_metrics


def run_experiment(exp_key, exp_config, seeds=range(1, 26)):
    """Run a single experiment configuration across multiple seeds."""
    
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_config['name']}")
    print(f"Description: {exp_config['description']}")
    print("="*80)
    
    results = []
    T = list(range(1, 29))
    K = [1, 2, 3]
    n_workers = exp_config['n_workers']
    I = list(range(1, n_workers + 1))
    
    data = pd.DataFrame({
        'I': I + [np.nan] * (max(len(I), len(T), len(K)) - len(I)),
        'T': T + [np.nan] * (max(len(I), len(T), len(K)) - len(T)),
        'K': K + [np.nan] * (max(len(I), len(T), len(K)) - len(K))
    })
    
    # Create worker groups
    if exp_config['team'] == 'mixed':
        worker_groups = create_groups_from_fractions(
            I, exp_config['fractions'], exp_config['group_params']
        )
    else:
        eps, chi = exp_config['group_params'][0]
        worker_groups = create_homogeneous_group(I, eps, chi)
    
    print(f"Worker groups: {[(g.name, len(g.worker_ids), g.epsilon, g.chi) for g in worker_groups.values()]}")
    
    for seed in seeds:
        print(f"\n--- Seed {seed}/{max(seeds)} ---")
        
        # Generate demand with scaling
        demand_dict = generate_dict_from_excel(
            'data/demand_scenarios.xlsx', len(I), 'Medium', scenario=seed
        )
        
        # Scale demand by Q_scar
        Q_scar = exp_config['Q_scar']
        if Q_scar != 1.0:
            demand_dict = {k: int(v * Q_scar) for k, v in demand_dict.items()}
        
        # Get representative epsilon for the run
        eps = exp_config['group_params'][0][0]
        chi = exp_config['group_params'][0][1]
        
        # Run CG with behavior
        t0 = time.time()
        try:
            (undercoverage_behavior, understaffing_behavior, perfloss_behavior, 
             consistency_behavior, consistency_norm_behavior, undercoverage_norm_behavior, 
             understaffing_norm_behavior, perfloss_norm_behavior, final_obj_behavior, 
             final_lb, itr, lagrangeB, gap, time_sps, time_rmp, time_ip, 
             ls_p_behavior, ls_sc_behavior, ls_perf_behavior, ls_x_behavior,
             ls_r_behavior, undercoverage_per_shift_behavior, results_ineq_sc_behavior, 
             spread_sc_behavior, load_share_sc_behavior, gini_sc_behavior, 
             results_ineq_perf_behavior, spread_perf_behavior, load_share_perf_behavior, 
             gini_perf_behavior, shift_blocks_behavior) = column_generation_behavior(
                data, demand_dict, eps, Min_WD_i, Max_WD_i, time_cg_init, max_itr, 
                output_len, chi, threshold, time_cg, I, T, K, Q_scar, 
                sp_solver='labeling_bidir', worker_groups=worker_groups, 
                save_lp=False, use_heuristic_start=True
            )
            
            run_time = time.time() - t0
            
            # Calculate per-group metrics
            group_metrics = calculate_group_metrics(ls_sc_behavior, worker_groups, len(T))
            
            result = {
                'experiment': exp_key,
                'seed': seed,
                'Q_scar': Q_scar,
                'team_type': exp_config['team'],
                'n_workers': n_workers,
                'undercoverage': undercoverage_behavior,
                'understaffing': understaffing_behavior,
                'consistency': consistency_behavior,
                'perfloss': perfloss_behavior,
                'objective': final_obj_behavior,
                'lb': final_lb,
                'gap': gap,
                'iterations': itr,
                'time': run_time,
                'group_metrics': group_metrics,
            }
            
            # Add per-group shift changes
            for g_name, g_metrics in group_metrics.items():
                result[f'sc_{g_name}'] = g_metrics['total_shift_changes']
                result[f'sc_avg_{g_name}'] = g_metrics['avg_shift_changes']
            
            results.append(result)
            print(f"  obj={final_obj_behavior:.2f}, gap={gap:.2f}%, iter={itr}, time={run_time:.1f}s")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    return results


def run_statistical_analysis(all_results):
    """Perform statistical tests for H1 and H3."""
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    # H1: Mann-Whitney U test for shift changes G1 vs G3
    if 'basis' in all_results:
        basis_results = all_results['basis']
        g1_sc = [r.get('sc_avg_group_1', 0) for r in basis_results]
        g3_sc = [r.get('sc_avg_group_3', 0) for r in basis_results]
        
        if g1_sc and g3_sc:
            stat, p_value = stats.mannwhitneyu(g1_sc, g3_sc, alternative='less')
            print(f"\n[H1] Mann-Whitney U Test: Shift Changes G1 vs G3")
            print(f"  G1 (Highly Sensitive) mean: {np.mean(g1_sc):.2f}")
            print(f"  G3 (Resilient) mean: {np.mean(g3_sc):.2f}")
            print(f"  U-statistic: {stat:.2f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Significant (p<0.05): {'YES ✓' if p_value < 0.05 else 'NO'}")
    
    # H3: Paired t-test for undercoverage (homogeneous vs mixed)
    if 'basis' in all_results and 'reference' in all_results:
        mixed_uc = [r['undercoverage'] for r in all_results['basis']]
        homo_uc = [r['undercoverage'] for r in all_results['reference']]
        
        # Match by seed
        min_len = min(len(mixed_uc), len(homo_uc))
        mixed_uc = mixed_uc[:min_len]
        homo_uc = homo_uc[:min_len]
        
        if mixed_uc and homo_uc:
            stat, p_value = stats.ttest_rel(homo_uc, mixed_uc)
            print(f"\n[H3] Paired t-Test: Undercoverage Homogeneous vs Mixed")
            print(f"  Homogeneous mean: {np.mean(homo_uc):.2f} (std: {np.std(homo_uc):.2f})")
            print(f"  Mixed mean: {np.mean(mixed_uc):.2f} (std: {np.std(mixed_uc):.2f})")
            print(f"  t-statistic: {stat:.2f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Mixed is better: {'YES ✓' if np.mean(mixed_uc) < np.mean(homo_uc) else 'NO'}")
            print(f"  Significant (p<0.05): {'YES ✓' if p_value < 0.05 else 'NO'}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    # Select experiments to run
    if RUN_EXPERIMENTS == 'all':
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = RUN_EXPERIMENTS
    
    print(f"Running experiments: {experiments_to_run}")
    print(f"Seeds per experiment: {N_SEEDS}")
    print(f"Total runs: {len(experiments_to_run) * N_SEEDS}")
    
    all_results = {}
    
    for exp_key in experiments_to_run:
        exp_config = EXPERIMENTS[exp_key]
        results = run_experiment(exp_key, exp_config, seeds=range(1, N_SEEDS + 1))
        all_results[exp_key] = results
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_excel(f'results/heterogeneity_{exp_key}_{datetime.now().strftime("%d_%m_%Y_%H-%M")}.xlsx', index=False)
    
    # Run statistical analysis
    run_statistical_analysis(all_results)
    
    # Save combined results
    all_flat = []
    for exp_key, results in all_results.items():
        all_flat.extend(results)
    
    df_all = pd.DataFrame(all_flat)
    df_all.to_excel(f'results/heterogeneity_ALL_{datetime.now().strftime("%d_%m_%Y_%H-%M")}.xlsx', index=False)
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    print("="*80)
