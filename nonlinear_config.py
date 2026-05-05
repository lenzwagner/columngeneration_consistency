experiment_config = {
    "instances": ["m100"],
    "scenario_ids": [0],
    "lambdas": [1.00],
    "model_specs": [
        {"family": "linear", "name": "LINEAR_BASE"},
        {"family": "nonlinear", "name": "NL_CONVEX"},
    ],
    "solver": {
        "time_limit": 7200,
        "mip_gap": 0.01,
        "threads": 1,
        "seed": 42,
        "rc_tol": 1e-5,
    }
}

nonlinear_specs = {
    "NL_CONVEX": {
        "name": "NL_CONVEX",
        "fatigue_levels": 8,
        "stability_levels": 8,
        "chg_mode": "convex",
        "rec_mode": "linear",
        "perf_mode": "linear",
        "transition_weights": {
            "same": 0.0,
            "mild": 1.0,
            "severe": 2.0,
            "return": 0.5,
        }
    },
    "NL_CONCAVE": {
        "name": "NL_CONCAVE",
        "fatigue_levels": 8,
        "stability_levels": 8,
        "chg_mode": "concave",
        "rec_mode": "linear",
        "perf_mode": "linear",
        "transition_weights": {
            "same": 0.0,
            "mild": 1.0,
            "severe": 2.0,
            "return": 0.5,
        }
    },
    "NL_SLOW_RECOVERY": {
        "name": "NL_SLOW_RECOVERY",
        "fatigue_levels": 8,
        "stability_levels": 8,
        "chg_mode": "linear",
        "rec_mode": "slow",
        "perf_mode": "linear",
        "transition_weights": {
            "same": 0.0,
            "mild": 1.0,
            "severe": 2.0,
            "return": 0.5,
        }
    },
    "NL_THRESHOLD_PERF": {
        "name": "NL_THRESHOLD_PERF",
        "fatigue_levels": 8,
        "stability_levels": 8,
        "chg_mode": "linear",
        "rec_mode": "linear",
        "perf_mode": "threshold",
        "transition_weights": {
            "same": 0.0,
            "mild": 1.0,
            "severe": 2.0,
            "return": 0.5,
        }
    }
}
