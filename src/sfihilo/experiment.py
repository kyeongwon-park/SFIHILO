from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import t as student_t

from . import core
from .labels import method_spec


RESULT_SLOTS = {
    "err_test": 0,
    "err_balanced": 1,
    "forward_rmse": 2,
    "policy_reward_mc": 3,
    "policy_regret_mc": 4,
    "policy_reward_mean": 5,
    "policy_regret_mean": 6,
    "current_err_test": 7,
    "current_err_balanced": 8,
}

SAVED_RESULT_KEYS = (
    "err_test",
    "err_balanced",
    "forward_rmse",
    "policy_reward_mc",
    "policy_regret_mc",
    "current_err_test",
    "current_err_balanced",
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_if_present(config: dict, key: str, target_name: str | None = None):
    if key in config:
        setattr(core, target_name or key, config[key])


def apply_runtime_overrides(config: dict) -> None:
    """Apply config values to the numerical constants used inside core.py."""
    for key in [
        "TARGET_KEPT",
        "INVERSE_GP_RFF_M",
        "INVERSE_GP_RBF_ELL",
        "INVERSE_GP_PRIOR_STD",
        "INVERSE_GP_ETA_MU",
        "INVERSE_GP_ETA_SIG",
        "INVERSE_GP_MAP_LBFGS_ITERS",
        "INVERSE_GP_POST_SAMPLES",
        "INVERSE_GP_COV_JITTER",
        "BOUNDARY_LOOKAHEAD_TARGET_TOPK",
        "BOUNDARY_LOOKAHEAD_SHORTLIST_SOFT",
        "BOUNDARY_LOOKAHEAD_SHORTLIST_VARIANCE",
        "BOUNDARY_LOOKAHEAD_SHORTLIST_RANDOM",
        "BOUNDARY_LOOKAHEAD_CANDIDATE_FORWARD_SAMPLES",
        "BOUNDARY_LOOKAHEAD_TARGET_FORWARD_SAMPLES",
        "BOUNDARY_LOOKAHEAD_SHORTLIST_KEEP",
        "BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_SOFT_EXTRA",
        "BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_VARIANCE_EXTRA",
        "BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_RANDOM_EXTRA",
        "JOINT_FWD_SAMPLES",
        "FORWARD_FULLKRR_VAL_FRAC",
        "FORWARD_FULLKRR_ELL_LIST",
        "FORWARD_FULLKRR_LAM_LIST",
        "FORWARD_FULLKRR_ELL_LIST_D2",
        "FORWARD_FULLKRR_LAM_LIST_D2",
        "FORWARD_FULLKRR_JITTER",
        "FORWARD_FULLKRR_TUNE_MAX_POINTS",
        "FORWARD_FULLKRR_WARMUP_SAMPLES",
        "N_CAL_X",
        "N_CAL_PHI",
        "GP_PHI_RFF_M",
        "GP_R_RFF_M",
        "POLICY_EVAL_BATCH_DEFAULT",
        "POLICY_MC_FWD_SAMPLES_DEFAULT",
        "FORWARD_TPE_SCHEDULE_N_TRIALS",
        "FORWARD_TPE_SCHEDULE_STARTUP_TRIALS",
        "FORWARD_TPE_SCHEDULE_TAU_RANGE",
        "FORWARD_TPE_SCHEDULE_H_RANGE",
    ]:
        _set_if_present(config, key)

    _set_if_present(config, "target_kept", "TARGET_KEPT")
    _set_if_present(config, "balanced_test_total", "GENERIC_BALANCED_TOTAL")
    _set_if_present(config, "min_outcome_axis_gap", "MIN_OUTCOME_AXIS_GAP_TAU_AND_GENERIC_B")

    metrics = config.get("metrics", {})
    if metrics:
        core.ENABLE_NONCORE_METRICS = bool(metrics.get("enable_noncore", core.ENABLE_NONCORE_METRICS))
        core.ENABLE_EXTRA_PREF_STATS = bool(metrics.get("extra_pref_stats", core.ENABLE_NONCORE_METRICS))
        core.ENABLE_GP_DIAGNOSTICS = bool(metrics.get("gp_diagnostics", core.ENABLE_NONCORE_METRICS))
        core.ENABLE_SHORTLIST_SOURCE_LOGGING = bool(metrics.get("shortlist_source_logging", True))
        core.ENABLE_QUERY_DIVERSITY_METRICS = bool(metrics.get("query_diversity", core.ENABLE_NONCORE_METRICS))
        core.ENABLE_TARGET_QUERY_METRIC = bool(metrics.get("target_query_metric", core.ENABLE_NONCORE_METRICS))


def _qc_list_for(config: dict, d_value: int) -> list[int]:
    qc = config.get("qc_list", [10, 30, 100])
    if isinstance(qc, dict):
        qc = qc.get(str(d_value), qc.get("default", [10, 30, 100]))
    return [int(x) for x in qc]


def _safe_key(value) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    keep = []
    for ch in text:
        keep.append(ch if (ch.isalnum() or ch == "_") else "_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "condition"


def _condition_specs(config: dict, d_list: list[int]) -> list[dict]:
    raw_conditions = config.get("conditions", None)
    if raw_conditions:
        specs = []
        for idx, raw in enumerate(raw_conditions):
            cond = dict(raw)
            d_value = int(cond.get(
                "d",
                cond.get("dimension", cond.get("total_outcome_dim", d_list[0] if d_list else 8)),
            ))
            key = _safe_key(cond.get("key", cond.get("name", f"condition_{idx}")))
            cond["key"] = key
            cond["label"] = str(cond.get("label", key))
            cond["d"] = d_value
            specs.append(cond)
        return specs

    return [
        {
            "key": "",
            "label": "",
            "type": "",
            "n_misspecified": "",
            "d": int(d_value),
        }
        for d_value in d_list
    ]


def _condition_config(config: dict, condition: dict) -> dict:
    merged = dict(config)
    merged.update(condition)
    return merged


def _seed_for(config: dict, d_value: int, run_idx: int) -> int:
    seed_base = int(config.get("seed_base", 1000))
    if bool(config.get("paired_world_by_run", False)):
        return seed_base + int(run_idx)
    if "seed_offset" in config:
        return seed_base + int(config["seed_offset"]) + int(run_idx)
    return seed_base + int(d_value) * 10_000 + int(run_idx)


def _initial_control_pair_for(control_dim: int, seed: int, offset: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed) + int(offset))
    return rng.rand(2, int(control_dim)).astype(np.float32)


def _control_pair_bank_for(control_dim: int, n_pairs: int, seed: int, offset: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed) + int(offset))
    xa = rng.rand(int(n_pairs), int(control_dim)).astype(np.float32)
    xb = rng.rand(int(n_pairs), int(control_dim)).astype(np.float32)
    return xa, xb


def _learner_reward_model_name(config: dict, method: dict) -> str:
    model = method.get("learner_reward_model", config.get("learner_reward_model", "gp_rff"))
    model = str(model).strip().lower()
    if model in ("gp", "gp_rff", "rff_gp"):
        return "gp_rff"
    if model in ("linear", "linear_laplace"):
        return "linear_laplace"
    raise ValueError("learner_reward_model must be 'gp_rff' or 'linear_laplace'.")


def _learner_reward_config(config: dict, method: dict, observed_dim: int) -> dict:
    model = _learner_reward_model_name(config, method)
    return {
        "model_type": model,
        "d_est": int(observed_dim),
        "prior_std_lin": float(config.get("learner_reward_prior_std_lin", 1.0)),
        "phi_mean": np.full((int(observed_dim),), 0.5, dtype=np.float64),
        "phi_std": np.full((int(observed_dim),), np.sqrt(1.0 / 12.0), dtype=np.float64),
    }


def _summary_context(*, config: dict, condition: dict, d_value: int,
                     method_label: str, learner_reward_model: str,
                     total_dim: int, observed_dim: int, reward_dim: int) -> dict:
    return {
        "experiment": str(config.get("experiment_kind", "")),
        "condition": str(condition.get("key", "")),
        "condition_label": str(condition.get("label", "")),
        "condition_type": str(condition.get("type", "")),
        "n_misspecified": condition.get("n_misspecified", ""),
        "d": int(d_value),
        "total_outcome_dim": int(total_dim),
        "observed_outcome_dim": int(observed_dim),
        "reward_outcome_dim": int(reward_dim),
        "learner_reward_model": str(learner_reward_model),
        "method": method_label,
    }


def _result_prefix(d_value: int, condition: dict, method: dict) -> str:
    parts = [f"d{int(d_value)}"]
    if condition.get("key", ""):
        parts.append(_safe_key(condition["key"]))
    parts.append(_safe_key(method["key"]))
    return "_".join(parts)


def _mean_ci(values: np.ndarray) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return np.nan, np.nan, 0
    mean = float(np.mean(arr))
    if n <= 1:
        return mean, 0.0, n
    ci = float(student_t.ppf(0.975, n - 1) * np.std(arr, ddof=1) / np.sqrt(n))
    return mean, ci, n


def _stack_results(results: list[tuple]) -> dict[str, np.ndarray]:
    stacked = {}
    for key, idx in RESULT_SLOTS.items():
        vals = [r[idx] for r in results]
        if key == "err_test":
            stacked[key] = np.stack(vals, axis=-1)
        else:
            stacked[key] = np.stack(vals, axis=-1)
    return stacked


def _append_summary_rows(rows: list[dict], *, context: dict,
                         qcs: np.ndarray, tau_list: list[float], stacked: dict[str, np.ndarray]) -> None:
    scalar_metrics = [
        ("preference_error_balanced", stacked["err_balanced"]),
        ("forward_rmse", stacked["forward_rmse"]),
        ("policy_reward_mc", stacked["policy_reward_mc"]),
        ("policy_regret_mc", stacked["policy_regret_mc"]),
    ]
    for metric, arr in scalar_metrics:
        for qi, qc in enumerate(qcs):
            mean, ci, n = _mean_ci(arr[qi])
            rows.append({
                **context,
                "metric": metric,
                "tau": "",
                "qc": int(qc),
                "mean": mean,
                "ci95": ci,
                "n": n,
            })

    err_test = stacked["err_test"]
    for ti, tau_val in enumerate(tau_list):
        for qi, qc in enumerate(qcs):
            mean, ci, n = _mean_ci(err_test[qi, ti, :])
            rows.append({
                **context,
                "metric": "preference_error_reward_gap_bin",
                "tau": tau_val,
                "qc": int(qc),
                "mean": mean,
                "ci95": ci,
                "n": n,
            })

    current_err_balanced = stacked.get("current_err_balanced")
    if current_err_balanced is not None and np.isfinite(current_err_balanced).any():
        for qi, qc in enumerate(qcs):
            mean, ci, n = _mean_ci(current_err_balanced[qi])
            rows.append({
                **context,
                "metric": "current_preference_error_balanced",
                "tau": "",
                "qc": int(qc),
                "mean": mean,
                "ci95": ci,
                "n": n,
            })

    current_err_test = stacked.get("current_err_test")
    if current_err_test is not None and np.isfinite(current_err_test).any():
        for ti, tau_val in enumerate(tau_list):
            for qi, qc in enumerate(qcs):
                mean, ci, n = _mean_ci(current_err_test[qi, ti, :])
                rows.append({
                    **context,
                    "metric": "current_preference_error_reward_gap_bin",
                    "tau": tau_val,
                    "qc": int(qc),
                    "mean": mean,
                    "ci95": ci,
                    "n": n,
                })


def _make_reward_oracle(config: dict, reward_dim: int, control_dim: int,
                        outcome_model, seed: int):
    reward_model = str(config.get("oracle_reward_model", config.get("reward_model", "gp"))).strip().lower()
    if reward_model in ("gp", "rff_gp", "simulated_gp"):
        return core.SimulatedRewardOracle(reward_dim, control_dim, outcome_model, seed=seed)
    raise ValueError("oracle_reward_model must be 'gp' for the reported experiments.")


def _build_one_task(config: dict, *, d_value: int, run_idx: int, method: dict,
                    phi_pairs_by_tau, phi_pairs_tau_balanced,
                    init_q, tda, tdb, noise_cfg, reward_oracle, outcome_model,
                    forward_model, policy_eval_pack, reward_cfg, seed: int,
                    total_dim: int, control_dim: int,
                    est_idx: list[int], true_idx: list[int]) -> tuple:
    qc_list = _qc_list_for(config, d_value)
    tau_list = [float(x) for x in config.get("tau_list", [0.02, 0.04, 0.08, 0.16])]
    candidate_pool_size = int(config.get("candidate_pool_size", core.compute_candidate_pool_size(control_dim)))
    return (
        qc_list,
        tau_list,
        phi_pairs_by_tau,
        phi_pairs_tau_balanced,
        None,
        init_q,
        tda,
        tdb,
        control_dim,
        noise_cfg,
        reward_oracle,
        est_idx,
        true_idx,
        0,
        np.eye(1),
        outcome_model,
        forward_model,
        int(seed),
        int(total_dim),
        int(candidate_pool_size),
        int(config.get("n_test_rmse", 1000)),
        reward_cfg,
        int(config.get("n_use_eval", 256)),
        True,
        True,
        method["query_policy"],
        method["active_mode"],
        float(config.get("target_error_rate", 0.10)),
        float(config.get("target_error_trigger", -0.30)),
        int(config.get("target_error_k", 3)),
        str(method.get("forward_weight_mode", config.get("forward_weight_mode", "none"))),
        bool(method.get("use_oracle_phi_for_acquisition", config.get("use_oracle_phi_for_acquisition", False))),
        policy_eval_pack,
    )


def run_experiment(config: dict) -> dict:
    apply_runtime_overrides(config)
    torch.set_num_threads(int(config.get("torch_num_threads", 1)))
    try:
        torch.set_num_interop_threads(int(config.get("torch_num_interop_threads", 1)))
    except RuntimeError:
        pass

    raw_output_dir = Path(config.get("output_dir", "results"))
    output_dir = raw_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_overrides = config.get("noise", None)
    effective_noise = dict(core.NOISE_DEFAULTS)
    if noise_overrides:
        effective_noise.update(noise_overrides)

    metadata_config = dict(config)
    metadata_config["noise"] = dict(effective_noise)
    metadata_config["output_dir"] = str(raw_output_dir if not raw_output_dir.is_absolute() else Path(raw_output_dir.name))

    d_list = [int(x) for x in config.get("d_list", [8])]
    conditions = _condition_specs(config, d_list)
    methods = [method_spec(name) for name in config.get(
        "methods", ["random", "mutual_information", "boundary_lookahead"]
    )]
    n_runs = int(config.get("n_runs", 30))
    tau_list = [float(x) for x in config.get("tau_list", [0.02, 0.04, 0.08, 0.16])]

    drift_dict = dict(config.get("drift", {"enabled": False}))
    drift_cfg = core.NonstationaryConfig(**{**core.DRIFT_DEFAULTS, **drift_dict})

    all_npz = {}
    summary_rows = []
    metadata = {
        "config": metadata_config,
        "methods": [
            {
                "key": m["key"],
                "label": m["label"],
                "query_policy": m["query_policy"],
                "active_mode": m["active_mode"],
                "forward_weight_mode": str(m.get("forward_weight_mode", config.get("forward_weight_mode", "none"))),
                "use_oracle_phi_for_acquisition": bool(m.get(
                    "use_oracle_phi_for_acquisition",
                    config.get("use_oracle_phi_for_acquisition", False),
                )),
                "learner_reward_model": _learner_reward_model_name(config, m),
            }
            for m in methods
        ],
        "conditions": conditions,
        "result_slots": {key: RESULT_SLOTS[key] for key in SAVED_RESULT_KEYS},
    }

    for condition in conditions:
        run_config = _condition_config(config, condition)
        d_value = int(condition["d"])
        control_dim = int(run_config.get("control_dim", d_value))
        reward_dim = int(run_config.get("reward_outcome_dim", d_value))
        observed_dim = int(run_config.get("observed_outcome_dim", d_value))
        total_dim = int(run_config.get("total_outcome_dim", max(reward_dim, observed_dim)))
        if reward_dim > total_dim or observed_dim > total_dim:
            raise ValueError("reward_outcome_dim and observed_outcome_dim must be <= total_outcome_dim.")
        true_idx = list(range(reward_dim))
        est_idx = list(range(observed_dim))

        qc_list = _qc_list_for(run_config, d_value)
        qcs = np.asarray([0] + qc_list, dtype=int)
        tasks_by_method = {m["key"]: [] for m in methods}
        paired_world = bool(run_config.get("paired_world_by_run", False))
        test_pool_size = int(run_config.get("test_pool_size", 200_000))
        test_pool_batch = int(run_config.get("test_pool_batch", 8192))
        initial_pair_seed_offset = int(run_config.get("initial_control_pair_seed_offset", 70_001))
        test_pair_seed_offset = int(run_config.get("test_control_pair_seed_offset", 70_011))

        for run_idx in range(n_runs):
            seed = _seed_for(run_config, d_value, run_idx)
            np.random.seed(seed)
            torch.manual_seed(seed)

            noise_cfg = core.NoiseConfig(base_seed=seed, overrides=noise_overrides)
            outcome_model, reward_oracle = None, None
            base_outcome_model = core.SimulatedOutcomeModel(control_dim, total_dim, seed=seed)
            base_outcome_model.eval()
            base_reward_oracle = _make_reward_oracle(run_config, reward_dim, control_dim, base_outcome_model, seed)
            if hasattr(base_reward_oracle, "eval"):
                base_reward_oracle.eval()
            outcome_model, reward_oracle = core.maybe_wrap_with_drift(
                base_outcome_model, base_reward_oracle, drift_cfg, true_idx=true_idx, seed=seed
            )
            if hasattr(outcome_model, "eval"):
                outcome_model.eval()

            initial_x_pair = (
                _initial_control_pair_for(control_dim, seed, initial_pair_seed_offset)
                if paired_world else None
            )
            init_q, tda, tdb = core.generate_initial_query(
                reward_oracle, control_dim, total_dim, observed_dim,
                noise_cfg, outcome_model, est_idx, true_idx,
                initial_x_pair=initial_x_pair,
            )

            clean_noise = core.NoiseConfig(
                base_seed=seed,
                overrides=dict(
                    control_jitter_std=0.0,
                    oracle_phi_noise_std=0.0,
                    reward_noise_std=0.0,
                ),
            )
            test_control_pair_bank = (
                _control_pair_bank_for(control_dim, test_pool_size, seed, test_pair_seed_offset)
                if paired_world else None
            )
            phi_pairs_by_tau, phi_pairs_tau_balanced, _ = core.build_nested_phi_pairs_by_tau(
                base_outcome_model,
                control_dim,
                est_idx,
                true_idx,
                base_reward_oracle,
                noise_cfg_clean=clean_noise,
                tau_list=tau_list,
                target_kept=int(run_config.get("target_kept", core.TARGET_KEPT)),
                pool_size=test_pool_size,
                batch_size=test_pool_batch,
                balanced_test_per_bin=int(run_config.get("balanced_test_per_bin", 250)),
                balanced_test_total=(
                    None if run_config.get("balanced_test_total", run_config.get("target_kept", core.TARGET_KEPT)) is None
                    else int(run_config.get("balanced_test_total", run_config.get("target_kept", core.TARGET_KEPT)))
                ),
                min_outcome_axis_gap=(
                    None if run_config.get("min_outcome_axis_gap", 0.05) is None
                    else float(run_config.get("min_outcome_axis_gap", 0.05))
                ),
                control_pair_bank=test_control_pair_bank,
                axis_gap_idx=(true_idx if paired_world else None),
            )

            policy_eval_pack = core.build_policy_eval_pack(
                outcome_model,
                reward_oracle,
                control_dim,
                true_idx,
                n_pool=int(run_config.get("policy_eval_pool_size", core.POLICY_EVAL_POOL_SIZE_DEFAULT)),
                seed=seed + 56000,
                batch_size=int(run_config.get("policy_eval_batch", core.POLICY_EVAL_BATCH_DEFAULT)),
                mc_samples=int(run_config.get("policy_mc_forward_samples", core.POLICY_MC_FWD_SAMPLES_DEFAULT)),
                mc_variance_key=str(run_config.get("policy_mc_variance_key", core.POLICY_MC_VARIANCE_KEY_DEFAULT)),
            )

            for method in methods:
                forward_model = core.build_forward_model(control_dim, observed_dim, seed=seed)
                reward_cfg = _learner_reward_config(run_config, method, observed_dim)
                tasks_by_method[method["key"]].append(_build_one_task(
                    run_config,
                    d_value=d_value,
                    run_idx=run_idx,
                    method=method,
                    phi_pairs_by_tau=phi_pairs_by_tau,
                    phi_pairs_tau_balanced=phi_pairs_tau_balanced,
                    init_q=init_q,
                    tda=tda,
                    tdb=tdb,
                    noise_cfg=copy.deepcopy(noise_cfg),
                    reward_oracle=reward_oracle,
                    outcome_model=outcome_model,
                    forward_model=forward_model,
                    policy_eval_pack=policy_eval_pack,
                    reward_cfg=reward_cfg,
                    seed=seed,
                    total_dim=total_dim,
                    control_dim=control_dim,
                    est_idx=est_idx,
                    true_idx=true_idx,
                ))

        n_jobs = int(config.get("n_jobs", 1))
        for method in methods:
            learner_reward_model = _learner_reward_model_name(run_config, method)
            condition_label = condition.get("label") or condition.get("key") or ""
            condition_text = f", condition={condition_label}" if condition_label else ""
            print(
                f"Running d={d_value}{condition_text}, method={method['label']}, "
                f"learner_reward_model={learner_reward_model}, runs={n_runs}"
            )
            if n_jobs == 1:
                results = [core.compute_metrics_multi_mode(task) for task in tasks_by_method[method["key"]]]
            else:
                with parallel_backend("loky", inner_max_num_threads=1):
                    results = Parallel(n_jobs=n_jobs)(
                        delayed(core.compute_metrics_multi_mode)(task)
                        for task in tasks_by_method[method["key"]]
                    )
            stacked = _stack_results(results)
            prefix = _result_prefix(d_value, condition, method)
            all_npz[f"{prefix}_qcs"] = qcs
            all_npz[f"{prefix}_tau_list"] = np.asarray(tau_list, dtype=np.float64)
            for key in SAVED_RESULT_KEYS:
                arr = stacked[key]
                if key.startswith("current_") and not np.isfinite(arr).any():
                    continue
                all_npz[f"{prefix}_{key}"] = np.asarray(arr)
            context = _summary_context(
                config=config,
                condition=condition,
                d_value=d_value,
                method_label=method["label"],
                learner_reward_model=learner_reward_model,
                total_dim=total_dim,
                observed_dim=observed_dim,
                reward_dim=reward_dim,
            )
            _append_summary_rows(
                summary_rows,
                context=context,
                qcs=qcs,
                tau_list=tau_list,
                stacked=stacked,
            )

    npz_path = output_dir / "results.npz"
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "metadata.json"
    np.savez_compressed(npz_path, **all_npz)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "condition",
                "condition_label",
                "condition_type",
                "n_misspecified",
                "d",
                "total_outcome_dim",
                "observed_outcome_dim",
                "reward_outcome_dim",
                "learner_reward_model",
                "method",
                "metric",
                "tau",
                "qc",
                "mean",
                "ci95",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {"npz": str(npz_path), "summary_csv": str(csv_path), "metadata_json": str(json_path)}
