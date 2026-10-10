# -*- coding: utf-8 -*-
from __future__ import annotations

# Keep numerical libraries single-threaded by default for reproducible CPU runs.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t

# =====================================================
# Core numerical settings
# =====================================================
# This supplementary implementation exposes experiment choices through
# JSON config files.  The constants below are defaults used by the runner.

# =====================================================
# Self-contained GP inverse knobs
# =====================================================
INVERSE_GP_RFF_M = 256
INVERSE_GP_RBF_ELL = 0.65
INVERSE_GP_PRIOR_STD = 1.0
INVERSE_GP_ETA_MU = 0.0
INVERSE_GP_ETA_SIG = 1.0
INVERSE_GP_MAP_LBFGS_ITERS = 50
INVERSE_GP_POST_SAMPLES = 512
INVERSE_GP_COV_JITTER = 1e-6

BOUNDARY_LOOKAHEAD_TARGET_TOPK = 32
BOUNDARY_LOOKAHEAD_SHORTLIST_SOFT = 16
BOUNDARY_LOOKAHEAD_SHORTLIST_VARIANCE = 8
BOUNDARY_LOOKAHEAD_SHORTLIST_RANDOM = 8
BOUNDARY_LOOKAHEAD_CANDIDATE_FORWARD_SAMPLES = 4
BOUNDARY_LOOKAHEAD_TARGET_FORWARD_SAMPLES = 3



# =====================================================
# Forward model: exact RBF kernel ridge regression
# =====================================================
# Alternative forward surrogates are intentionally not exposed in this
# supplementary package. Full KRR is the forward model used by the
# reported Boundary Lookahead experiments.
FORWARD_FULLKRR_VAL_FRAC         = 0.20
FORWARD_FULLKRR_ELL_LIST         = [0.3, 0.4, 0.5, 0.6, 0.7]
FORWARD_FULLKRR_LAM_LIST         = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
FORWARD_FULLKRR_ELL_LIST_D2      = [0.08, 0.10, 0.12, 0.15, 0.20]
FORWARD_FULLKRR_LAM_LIST_D2      = [1e-4, 1e-3, 3e-3, 1e-2]
FORWARD_FULLKRR_JITTER           = 1e-10
FORWARD_FULLKRR_TUNE_MAX_POINTS  = 1024
FORWARD_FULLKRR_WARMUP_SAMPLES   = 16

# Optional drift forward-learning HPO for estimated alpha schedules.
# Optimize (tau, h, ell, lambda), then train FullKRR with alpha(q; tau) weights.
FORWARD_TPE_SCHEDULE_N_TRIALS       = 64
FORWARD_TPE_SCHEDULE_STARTUP_TRIALS = 16
FORWARD_TPE_SCHEDULE_TAU_RANGE      = (15.0, 400.0)
FORWARD_TPE_SCHEDULE_H_RANGE        = (0.02, 0.40)

# =====================================================
# Robust active-query settings (forward uncertainty -> U distribution)
# =====================================================
# ROBUST_FWD_SAMPLES   = 16    # M: forward posterior samples per candidate
# ROBUST_TOP_FRAC      = 0.10  # keep top-10% candidates by mean U
# ROBUST_U_ALPHA       = 0.20  # conservative lower quantile q_alpha(U)
# ROBUST_POOL_CAP      = 1024  # cap pool size for robust scoring cost
ROBUST_FWD_SAMPLES   = 48    # M: forward posterior samples per candidate
ROBUST_TOP_FRAC      = 0.01  # keep top-10% candidates by mean U
ROBUST_U_ALPHA       = 0.05  # conservative lower quantile q_alpha(U)
ROBUST_POOL_CAP      = 2048  # cap pool size for robust scoring cost

# =====================================================
# Joint active-query settings (forward uncertainty + inverse uncertainty)
# =====================================================
JOINT_FWD_SAMPLES    = 8     # forward posterior samples per candidate for joint pool scoring
JOINT_LAMBDA_HALF    = 0.5   # default forward bonus multiplier used by *_lam05 modes

# =====================================================
# Auxiliary boundary-lookahead scoring modes
#   - cond_soft:           cheap Jensen-fix sanity check
#   - boundary_lookahead_soft:    soft boundary-target expected observation value
#   - boundary_lookahead_jointmi: joint-MI boundary-target expected observation value
# =====================================================
CONDITIONED_SOFT_FORWARD_SAMPLES            = 8
BOUNDARY_LOOKAHEAD_SOFT_TARGET_TOPK                = 32   # target boundary set size
BOUNDARY_LOOKAHEAD_SOFT_SHORTLIST_STARTS           = 32   # soft-opt proposal starts
BOUNDARY_LOOKAHEAD_SHORTLIST_KEEP             = 12   # additional top pool-soft shortlist budget
BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_SOFT_EXTRA  = 4    # add top soft-disagreement pool points
BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_VARIANCE_EXTRA   = 4    # add top forward-variance pool points
BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_RANDOM_EXTRA = 4   # add random pool points to final shortlist
BOUNDARY_LOOKAHEAD_SOFT_CANDIDATE_FORWARD_SAMPLES           = 4    # candidate x: forward posterior samples
BOUNDARY_LOOKAHEAD_SOFT_TARGET_FORWARD_SAMPLES         = 3    # target-set forward samples (current / post)
BOUNDARY_LOOKAHEAD_SOFTOPT_STEPS              = 60
BOUNDARY_LOOKAHEAD_SOFTOPT_LR                 = 0.03

JOINT_SELDIAG_LABELS = (
    "selected acquisition score",
    "selected main score",
    "selected forward bonus",
    "selected score multiplier",
    "selected pbar",
    "selected forward label distortion",
    "selected aux_fwd",
    "selected aux_inv",
    "selected target current uncertainty",
    "selected target expected post uncertainty",
    "selected target predicted uncertainty gain",
    "selected reward-only predicted gain",
    "selected forward epistemic reduction",
    "selected target oracle-gap current",
    "selected shortlist source: soft-opt",
    "selected shortlist source: pool-soft",
    "selected shortlist source: pool-var",
    "selected shortlist source: random",
)

GP_SELDIAG_SPECS = (
    ("selected acquisition score", "score"),
    ("selected predicted pbar", "pbar"),
    ("selected predicted soft score U=4p(1-p)", "soft_score"),
    ("selected predicted forward variance", "forward_var"),
    ("selected observed pbar", "observed_pbar"),
    ("selected observed soft score U=4p(1-p)", "observed_soft_score"),
    ("selected predicted target current utility", "target_cur"),
    ("selected predicted target expected post utility", "target_post"),
    ("selected predicted target utility gain", "target_gain"),
    ("selected shortlist source: opt", "shortlist_src_softopt"),
    ("selected shortlist source: pool-main", "shortlist_src_soft"),
    ("selected shortlist source: pool-var", "shortlist_src_var"),
    ("selected shortlist source: random", "shortlist_src_random"),
    ("selected shortlist size", "shortlist_size"),
    ("selected target set size", "target_size"),
)
GP_SELDIAG_LABELS = tuple(label for (label, _) in GP_SELDIAG_SPECS)

GP_DIAG_SPECS = (
    ("GP n_obs", 0),
    ("GP beta_mean", 1),
    ("GP beta_map", 2),
    ("GP eta_map", 3),
    ("GP cov_trace", 4),
    ("GP clipped_eigs_total", 5),
)

# ===== Robustness noise (reproducible) =====
NOISE_DEFAULTS = dict(
    control_jitter_std=0.02,
    oracle_phi_noise_std=0.05,
    reward_noise_std=0.02
)

DRIFT_DEFAULTS = dict(
    enabled=True,
    q95_queries=200.0,
    target_alpha=0.95,
    eval_pair_bank=50_000,
    eval_forward_bank=1000,
    forward_alpha_weight_bandwidth=0.15,
    forward_alpha_weight_floor=1e-4,
)

# =====================================================
# Test-set config (filtered pairs only)
# =====================================================
TARGET_KEPT     = 1000
MAX_DRAWS_MULT  = 20

# Final policy-selection metric.  The pool approximates the policy search
# space; MC integrates the learned forward belief for the main metric.
POLICY_EVAL_POOL_SIZE_DEFAULT = 50_000
POLICY_EVAL_BATCH_DEFAULT = 2048
POLICY_MC_FWD_SAMPLES_DEFAULT = 16
POLICY_MC_VARIANCE_KEY_DEFAULT = "var_epi"

# =====================================================
# Optional non-core metrics / diagnostics
#   Core metrics (always on):
#     - error rate
#     - RMSE
#   Non-core metrics below can be toggled independently.
# =====================================================
ENABLE_NONCORE_METRICS = False
ENABLE_EXTRA_PREF_STATS = ENABLE_NONCORE_METRICS       # reward-gap and objective-U summaries
ENABLE_GP_DIAGNOSTICS = ENABLE_NONCORE_METRICS       # GP posterior diagnostics
ENABLE_SHORTLIST_SOURCE_LOGGING = False                # selected shortlist-source diagnostics
ENABLE_QUERY_DIVERSITY_METRICS = ENABLE_NONCORE_METRICS  # dx/dphi novelty summaries
ENABLE_TARGET_QUERY_METRIC = False

# =====================================================
# Utils
# =====================================================
def set_printopts():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

def clip01(arr):
    return np.clip(arr, 0.0, 1.0)

def thin_samples_evenly(arr, n_max: int):
    S = arr.shape[0]
    if (n_max is None) or (n_max <= 0) or (S <= n_max):
        return arr
    idx = np.linspace(0, S-1, num=int(n_max), dtype=int)
    return arr[idx]

def _sigmoid(x):
    # stable-ish sigmoid
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _reward_gap_noise_std(noise_cfg=None) -> float:
    if noise_cfg is None:
        return 0.0
    reward_std = float(getattr(noise_cfg, 'reward_noise_std', 0.0))
    if (not np.isfinite(reward_std)) or reward_std <= 0.0:
        return 0.0
    return float(np.sqrt(2.0) * reward_std)


def _pref_probs_from_logits(logits, beta=None, *, noise_cfg=None):
    """Acquisition-side noisy-label probability approximation.

    The simulator compares two rewards with independent reward noise.  For
    acquisition scoring only, shrink beta*gap logits by a logistic-normal
    approximation. The learner likelihood is intentionally left unchanged.
    """
    logits = np.asarray(logits, dtype=np.float64)
    gap_noise = _reward_gap_noise_std(noise_cfg)
    if gap_noise > 0.0 and beta is not None:
        beta_arr = np.asarray(beta, dtype=np.float64)
        denom = np.sqrt(1.0 + (np.pi / 8.0) * (np.abs(beta_arr) * gap_noise) ** 2)
        logits = logits / np.maximum(denom, 1e-12)
    sigm = _sigmoid(logits)
    return sigm


def _reducible_variance_score(var_epi, var_total, axis=1):
    var_epi = np.maximum(np.asarray(var_epi, dtype=np.float64), 0.0)
    var_total = np.maximum(np.asarray(var_total, dtype=np.float64), 1e-8)
    return np.mean((var_epi * var_epi) / var_total, axis=axis)


def _reducible_variance_fraction(var_epi, var_total, axis=1):
    var_epi = np.maximum(np.asarray(var_epi, dtype=np.float64), 0.0)
    var_total = np.maximum(np.asarray(var_total, dtype=np.float64), 1e-8)
    return np.clip(np.mean(var_epi / var_total, axis=axis), 0.0, 1.0)


def forward_rmse_np(a: np.ndarray, b: np.ndarray, sample_weight=None) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sq = np.mean((a - b) ** 2, axis=1) if a.ndim > 1 else (a - b) ** 2
    if sample_weight is None:
        return float(np.sqrt(np.mean(sq)))
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.size != sq.shape[0]:
        return float(np.sqrt(np.mean(sq)))
    w = np.maximum(w, 0.0)
    wsum = float(np.sum(w))
    if (not np.isfinite(wsum)) or wsum <= 1e-12:
        return float(np.sqrt(np.mean(sq)))
    return float(np.sqrt(np.sum(w * sq) / wsum))


def standardize_fit(Y: np.ndarray, sample_weight=None):
    Y = np.asarray(Y, dtype=np.float64)
    if sample_weight is None:
        mu = np.mean(Y, axis=0)
        sd = np.std(Y, axis=0)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != Y.shape[0]:
            mu = np.mean(Y, axis=0)
            sd = np.std(Y, axis=0)
        else:
            w = np.maximum(w, 0.0)
            wsum = float(np.sum(w))
            if (not np.isfinite(wsum)) or wsum <= 1e-12:
                mu = np.mean(Y, axis=0)
                sd = np.std(Y, axis=0)
            else:
                wn = w / wsum
                mu = np.sum(wn[:, None] * Y, axis=0)
                var = np.sum(wn[:, None] * (Y - mu[None, :]) ** 2, axis=0)
                sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.maximum(sd, 1e-6)
    return mu, sd


def rbf_kernel_np(X: np.ndarray, Z: np.ndarray, ell: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Z2 = np.sum(Z * Z, axis=1, keepdims=True).T
    D2 = X2 + Z2 - 2.0 * (X @ Z.T)
    D2 = np.maximum(D2, 0.0)
    return np.exp(-0.5 * D2 / (float(ell) ** 2 + 1e-12))


def rbf_kernel_torch(X: torch.Tensor, Z: torch.Tensor, ell: float) -> torch.Tensor:
    X = torch.as_tensor(X, dtype=torch.float32, device=Z.device if torch.is_tensor(Z) else None)
    Z = torch.as_tensor(Z, dtype=torch.float32, device=X.device)
    X2 = torch.sum(X * X, dim=1, keepdim=True)
    Z2 = torch.sum(Z * Z, dim=1, keepdim=True).t()
    D2 = torch.clamp(X2 + Z2 - 2.0 * (X @ Z.t()), min=0.0)
    return torch.exp(-0.5 * D2 / (float(ell) ** 2 + 1e-12))

# -----------------------------
# Effective Sample Size
# -----------------------------
def _autocorr_fft_1d(x: np.ndarray):
    """Fast autocorrelation via FFT. Returns acf[0..N-1] normalized with acf[0]=1."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.ones(n, dtype=np.float64)
    x = x - x.mean()
    m = 1 << (2*n - 1).bit_length()
    fx = np.fft.rfft(x, n=m)
    ac = np.fft.irfft(fx * np.conj(fx), n=m)[:n]
    ac = ac / ac[0] if ac[0] > 0 else ac
    return ac

def ess_1d(x: np.ndarray):
    """
    ESS using Geyer's initial positive sequence on autocorrelation.
    Returns ESS (<=N). Robust-ish, no external deps.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        return float(n)
    ac = _autocorr_fft_1d(x)
    tau = 1.0
    for k in range(1, n-1, 2):
        pair = ac[k] + ac[k+1]
        if pair <= 0.0:
            break
        tau += 2.0 * pair
    if tau <= 0:
        return float(n)
    return float(n / tau)

def ess_summary(theta_samples: np.ndarray, max_dims=12, rng=None):
    """
    Return a robust ESS summary for theta:
      - choose a small subset of dims (first few or random)
      - return median ESS across chosen dims
    """
    if rng is None:
        rng = np.random
    S, D = theta_samples.shape
    if S < 5:
        return float(S)
    if D <= max_dims:
        dims = np.arange(D)
    else:
        head = np.arange(min(4, D))
        rest = rng.choice(np.arange(4, D), size=max_dims - head.size, replace=False)
        dims = np.concatenate([head, rest])
    ess_list = [ess_1d(theta_samples[:, d]) for d in dims]
    return float(np.median(ess_list))

# =====================================================
# (GP) — parameters & calibration
# =====================================================
X_LOW, X_HIGH = 0.0, 1.0

N_CAL_X   = 20000
N_CAL_PHI = 20000

GP_PHI_RFF_M        = 1024
GP_PHI_LEN_MIN      = 0.35
GP_PHI_LEN_MAX      = 0.65
GP_PHI_SHARED_OMEGA = False

GP_R_RFF_M          = 512
GP_R_LEN_MIN        = 0.50
GP_R_LEN_MAX        = 0.80

def rand_in_box(n, d, low=X_LOW, high=X_HIGH):
    return (np.random.rand(n, d).astype(np.float32) * (high - low) + low)

class MinMaxScaler01:
    def __init__(self):
        self.min_ = None
        self.range_ = None

    def fit(self, arr):
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        self.min_ = a.min(axis=0)
        max_ = a.max(axis=0)
        self.range_ = np.maximum(max_ - self.min_, 1e-12)
        return self

    def transform(self, arr):
        a = np.asarray(arr, dtype=np.float64)
        one_dim = False
        if a.ndim == 1:
            a = a[:, None]
            one_dim = True
        out = (a - self.min_[None, :]) / self.range_[None, :]
        out = np.clip(out, 0.0, 1.0)
        if one_dim:
            out = out[:, 0]
        return out

class _RFFHead(nn.Module):
    def __init__(self, in_dim, M, len_min, len_max, seed=None,
                 shared_Omega=None, shared_b=None):
        super().__init__()
        rs = np.random.RandomState(None if seed is None else int(seed))

        u = rs.uniform(0.0, 1.0)
        ell = float(np.exp(np.log(len_min) * (1 - u) + np.log(len_max) * u))

        if shared_Omega is None:
            Omega = rs.normal(0.0, 1.0, size=(M, in_dim)).astype(np.float32)
            self.register_buffer('Omega', torch.from_numpy(Omega))
        else:
            self.register_buffer('Omega', shared_Omega)

        if shared_b is None:
            b = rs.uniform(0.0, 2*np.pi, size=(M,)).astype(np.float32)
            self.register_buffer('b', torch.from_numpy(b))
        else:
            self.register_buffer('b', shared_b)

        w = rs.normal(0.0, 1.0, size=(M,)).astype(np.float32)
        self.register_buffer('w', torch.from_numpy(w))
        self.register_buffer('lengthscale', torch.tensor(ell, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(0.0, dtype=torch.float32))
        self.sqrt2_over_M = math.sqrt(2.0 / float(M))

    def forward(self, x):
        Z = (x / self.lengthscale) @ self.Omega.t()
        Z = torch.cos(Z + self.b) * self.sqrt2_over_M
        y = (Z * self.w).sum(dim=-1) * self.scale + self.bias
        return y

class GPMapRFF(nn.Module):
    def __init__(self, in_dim, out_dim,
                 M=1024, len_min=0.15, len_max=0.8,
                 shared=False, seed=None):
        super().__init__()
        rs = np.random.RandomState(None if seed is None else int(seed))
        if shared:
            Omega = rs.normal(0.0, 1.0, size=(M, in_dim)).astype(np.float32)
            b     = rs.uniform(0.0, 2*np.pi, size=(M,)).astype(np.float32)
            shared_Omega = torch.from_numpy(Omega)
            shared_b     = torch.from_numpy(b)
            self.heads = nn.ModuleList([
                _RFFHead(in_dim, M, len_min, len_max,
                         seed=None if seed is None else (int(seed) + 1000 + k),
                         shared_Omega=shared_Omega, shared_b=shared_b)
                for k in range(out_dim)
            ])
        else:
            self.heads = nn.ModuleList([
                _RFFHead(in_dim, M, len_min, len_max,
                         seed=None if seed is None else (int(seed) + 1000 + k))
                for k in range(out_dim)
            ])

    def forward(self, x):
        outs = [head(x).unsqueeze(-1) for head in self.heads]
        return torch.cat(outs, dim=-1)

class GPRewardRFF(_RFFHead):
    def __init__(self, dim, seed=None, M=GP_R_RFF_M,
                 len_min=GP_R_LEN_MIN, len_max=GP_R_LEN_MAX):
        super().__init__(dim, M, len_min, len_max, seed=seed)

# =====================================================
# Oracles
# =====================================================
class SimulatedOutcomeModel(nn.Module):
    def __init__(self, control_dim, outcome_dim_total, seed=None):
        super().__init__()
        self.control_dim = int(control_dim)
        self.outcome_dim_total = int(outcome_dim_total)
        self.gpphi = GPMapRFF(
            in_dim=self.control_dim, out_dim=self.outcome_dim_total,
            M=GP_PHI_RFF_M,
            len_min=GP_PHI_LEN_MIN, len_max=GP_PHI_LEN_MAX,
            shared=GP_PHI_SHARED_OMEGA,
            seed=seed
        )
        Xcal = rand_in_box(N_CAL_X, self.control_dim)
        with torch.no_grad():
            Phi_cal_raw = self.gpphi(torch.tensor(Xcal, dtype=torch.float32)).cpu().numpy().astype(np.float64)
        self.scaler_phi = MinMaxScaler01().fit(Phi_cal_raw)
        # --- add: torch buffers for differentiable scaling
        self.register_buffer("phi_min_t", torch.tensor(self.scaler_phi.min_, dtype=torch.float32))
        self.register_buffer("phi_range_t", torch.tensor(self.scaler_phi.range_, dtype=torch.float32))


    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        one = (x.dim() == 1)
        if one:
            x = x.unsqueeze(0)
        with torch.no_grad():
            phi_raw = self.gpphi(x)
        phi_np = phi_raw.detach().cpu().numpy()
        phi = self.scaler_phi.transform(phi_np).astype(np.float32)
        out = torch.tensor(phi, dtype=torch.float32, device=x.device)
        if one:
            out = out.squeeze(0)
        return out
    
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward for optimization.
        x: torch.Tensor (R, control_dim) on any device
        returns: scaled phi (R, outcome_dim_total) on same device
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # no torch.no_grad() here!
        phi_raw = self.gpphi(x)  # (R, outcome_dim_total), differentiable

        # min-max scaling with buffers (constants) - differentiable (except clamp boundary)
        phi_min = self.phi_min_t.to(device=x.device, dtype=phi_raw.dtype)
        phi_rng = self.phi_range_t.to(device=x.device, dtype=phi_raw.dtype)
        phi = (phi_raw - phi_min[None, :]) / (phi_rng[None, :] + 1e-12)
        phi = torch.clamp(phi, 0.0, 1.0)
        return phi


class SimulatedRewardOracle(nn.Module):
    def __init__(self, outcome_dim, control_dim_for_cal, outcome_model_for_cal: SimulatedOutcomeModel, seed=None):
        super().__init__()
        self.outcome_dim = int(outcome_dim)
        self.gpr = GPRewardRFF(
            dim=self.outcome_dim,
            seed=None if seed is None else (int(seed)+777),
            M=GP_R_RFF_M,
            len_min=GP_R_LEN_MIN, len_max=GP_R_LEN_MAX
        )
        Xcal2 = rand_in_box(N_CAL_PHI, int(control_dim_for_cal))
        with torch.no_grad():
            Phi_cal_full = outcome_model_for_cal(torch.tensor(Xcal2, dtype=torch.float32)).cpu().numpy().astype(np.float32)
        Phi_cal = Phi_cal_full[:, :self.outcome_dim]
        with torch.no_grad():
            r_cal_raw = self.gpr(torch.tensor(Phi_cal, dtype=torch.float32)).cpu().numpy().astype(np.float64)
        self.scaler_r = MinMaxScaler01().fit(r_cal_raw)

    def forward(self, phi_tensor):
        phi_tensor = torch.as_tensor(phi_tensor, dtype=torch.float32)
        one = (phi_tensor.dim() == 1)
        if one:
            phi_tensor = phi_tensor.unsqueeze(0)
        with torch.no_grad():
            r_raw = self.gpr(phi_tensor)
        r_np = r_raw.detach().cpu().numpy()
        r = self.scaler_r.transform(r_np).astype(np.float32)
        out = torch.tensor(r, dtype=torch.float32, device=phi_tensor.device)
        if one:
            out = out.squeeze(0)
        return out

# =====================================================
# Forward surrogate φ_est(x)
# =====================================================

class FullKRRForwardModel(nn.Module):
    """Exact/full RBF KRR forward surrogate.

    In this file, exact refit is done on every query update, while
    hyperparameter sweep is triggered only when the caller passes
    `retune=True` (wired to `j in qc_list` in the main loop).
    """

    def __init__(
        self,
        control_dim: int,
        outcome_dim_est: int,
        ell_list=None,
        lam_list=None,
        val_frac: float = FORWARD_FULLKRR_VAL_FRAC,
        jitter: float = FORWARD_FULLKRR_JITTER,
        tune_max_points: int = FORWARD_FULLKRR_TUNE_MAX_POINTS,
        warmup_samples: int = FORWARD_FULLKRR_WARMUP_SAMPLES,
        seed: int = 0,
        default_value: float = 0.5,
    ):
        super().__init__()
        self.control_dim = int(control_dim)
        self.outcome_dim_est = int(outcome_dim_est)
        if ell_list is None or lam_list is None:
            ell_list_default, lam_list_default = _forward_fullkrr_lists_for_control_dim(self.control_dim)
            if ell_list is None:
                ell_list = ell_list_default
            if lam_list is None:
                lam_list = lam_list_default
        self.ell_list = [float(v) for v in ell_list]
        self.lam_list = [float(v) for v in lam_list]
        if len(self.ell_list) <= 0 or len(self.lam_list) <= 0:
            raise ValueError("FORWARD_FULLKRR_ELL_LIST / FORWARD_FULLKRR_LAM_LIST must be non-empty.")
        self.val_frac = float(val_frac)
        self.jitter = float(jitter)
        self.tune_max_points = int(max(8, tune_max_points))
        self.warmup_samples = int(max(1, warmup_samples))
        self.seed = int(seed) if seed is not None else 0
        self.default_value = float(default_value)
        self._var_floor = 1e-4

        self.register_buffer("Xtr_buf", torch.empty(0, dtype=torch.float32))
        self.register_buffer("alpha_buf", torch.empty(0, dtype=torch.float32))
        self.register_buffer("y_mu_buf", torch.empty(0, dtype=torch.float32))
        self.register_buffer("y_sd_buf", torch.empty(0, dtype=torch.float32))
        self.register_buffer("chol_buf", torch.empty(0, dtype=torch.float32))
        self.register_buffer("emp_mean_buf", torch.full((self.outcome_dim_est,), float(self.default_value), dtype=torch.float32))
        self.register_buffer("emp_var_buf", torch.full((self.outcome_dim_est,), 0.05 ** 2, dtype=torch.float32))
        self.register_buffer("resid_var_buf", torch.full((self.outcome_dim_est,), 0.05 ** 2, dtype=torch.float32))

        self._X_hist = np.empty((0, self.control_dim), dtype=np.float32)
        self._Y_hist = np.empty((0, self.outcome_dim_est), dtype=np.float32)
        self._alpha_hist = np.empty((0,), dtype=np.float64)
        self._forward_weight_mode = "none"
        self._current_alpha_for_weight = None
        self._alpha_weight_bandwidth = 0.15
        self._alpha_weight_floor = 1e-4
        self._y_sum = np.zeros((self.outcome_dim_est,), dtype=np.float64)
        self._y_sq_sum = np.zeros((self.outcome_dim_est,), dtype=np.float64)

        self._n_seen = 0
        self._n_fit_seen = 0
        self._fit_count = 0
        self._last_tune_n = 0
        self._best_ell = self._nearest_value(self.ell_list, 0.65, log_scale=False)
        self._best_lam = self._nearest_value(self.lam_list, 1e-5, log_scale=True)
        self._is_fit = False

    @staticmethod
    def _nearest_value(values, target: float, log_scale: bool = False) -> float:
        arr = np.asarray(list(values), dtype=np.float64)
        if arr.size <= 0:
            raise ValueError("values must be non-empty")
        if log_scale:
            score = np.abs(np.log(np.maximum(arr, 1e-30)) - math.log(max(float(target), 1e-30)))
        else:
            score = np.abs(arr - float(target))
        return float(arr[int(np.argmin(score))])

    @property
    def n_seen(self) -> int:
        return int(self._n_seen)

    @property
    def n_fit_seen(self) -> int:
        return int(self._n_fit_seen)

    def _update_empirical_stats(self, Y_new: np.ndarray) -> None:
        if Y_new.size <= 0:
            return
        Y64 = np.asarray(Y_new, dtype=np.float64)
        self._y_sum += np.sum(Y64, axis=0)
        self._y_sq_sum += np.sum(Y64 * Y64, axis=0)
        n = int(self._n_seen)
        if n <= 0:
            mean = np.full((self.outcome_dim_est,), float(self.default_value), dtype=np.float32)
            var = np.full((self.outcome_dim_est,), 0.05 ** 2, dtype=np.float32)
        else:
            mean64 = self._y_sum / float(n)
            var64 = (self._y_sq_sum / float(n)) - mean64 * mean64
            var64 = np.maximum(var64, self._var_floor)
            mean = mean64.astype(np.float32)
            var = var64.astype(np.float32)
        with torch.no_grad():
            self.emp_mean_buf.copy_(torch.from_numpy(mean).to(device=self.emp_mean_buf.device, dtype=torch.float32))
            self.emp_var_buf.copy_(torch.from_numpy(var).to(device=self.emp_var_buf.device, dtype=torch.float32))

    def request_retune(self) -> None:
        """Mark that the next exact refit should re-sweep (ell, lam)."""
        # Kept as an explicit hook so retune policy is decided outside the model.
        # In this file, the caller uses qc checkpoints via `j in qc_list`.
        self._last_tune_n = int(self._n_seen)

    def _alpha_weights_for(self, alpha_values, current_alpha=None, normalize: bool = True) -> np.ndarray:
        alpha_arr = np.asarray(alpha_values, dtype=np.float64).reshape(-1)
        if alpha_arr.size <= 0 or not np.all(np.isfinite(alpha_arr)):
            return None
        alpha_cur = getattr(self, "_current_alpha_for_weight", None) if current_alpha is None else current_alpha
        if alpha_cur is None or not np.isfinite(float(alpha_cur)):
            return None
        h = max(float(getattr(self, "_alpha_weight_bandwidth", 0.15)), 1e-8)
        floor = max(float(getattr(self, "_alpha_weight_floor", 1e-4)), 0.0)
        dist = alpha_arr - float(alpha_cur)
        w = np.exp(-0.5 * (dist / h) ** 2)
        w = np.maximum(w, floor)
        if bool(normalize):
            mean_w = float(np.mean(w)) if w.size > 0 else 1.0
            if np.isfinite(mean_w) and mean_w > 1e-12:
                w = w / mean_w
        return np.asarray(w, dtype=np.float64)

    def _current_sample_weights(self) -> np.ndarray:
        mode = str(getattr(self, "_forward_weight_mode", "none")).lower().strip()
        if mode not in ("alpha", "alpha_weighted", "alpha-weighted"):
            return None
        if self._alpha_hist.shape[0] != int(self._n_seen):
            return None
        return self._alpha_weights_for(self._alpha_hist, normalize=True)

    def _fit_exact_np(self, Xtr: np.ndarray, Ytr: np.ndarray, ell: float, lam: float, sample_weight=None):
        Xtr64 = np.asarray(Xtr, dtype=np.float64)
        Ytr64 = np.asarray(Ytr, dtype=np.float64)
        w = None
        if sample_weight is not None:
            w_try = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if w_try.size == Xtr64.shape[0]:
                w = np.maximum(w_try, 1e-12)
        y_mu, y_sd = standardize_fit(Ytr64, sample_weight=w)
        Ys = (Ytr64 - y_mu) / y_sd
        K = rbf_kernel_np(Xtr64, Xtr64, ell)
        n = K.shape[0]
        if w is None:
            A = K + (float(lam) + float(self.jitter)) * np.eye(n, dtype=np.float64)
        else:
            A = K + (float(lam) + float(self.jitter)) * np.diag(1.0 / w)
        try:
            chol = np.linalg.cholesky(A)
            tmp = np.linalg.solve(chol, Ys)
            alpha = np.linalg.solve(chol.T, tmp)
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(A, Ys)
            chol = np.linalg.cholesky(A + 1e-8 * np.eye(n, dtype=np.float64))
        pred = (K @ alpha) * y_sd + y_mu
        if w is None:
            resid_var = np.maximum(np.mean((pred - Ytr64) ** 2, axis=0), self._var_floor)
        else:
            wsum = max(float(np.sum(w)), 1e-12)
            resid_var = np.maximum(np.sum(w[:, None] * (pred - Ytr64) ** 2, axis=0) / wsum, self._var_floor)
        return dict(Xtr=Xtr64, alpha=alpha, y_mu=y_mu, y_sd=y_sd, chol=chol, resid_var=resid_var)

    def _choose_tuning_subset(self):
        X = np.asarray(self._X_hist, dtype=np.float32)
        Y = np.asarray(self._Y_hist, dtype=np.float32)
        A = np.asarray(self._alpha_hist, dtype=np.float64).reshape(-1)
        if A.shape[0] != X.shape[0]:
            A = None
        n = X.shape[0]
        if n <= self.tune_max_points:
            return X, Y, A
        rs = np.random.RandomState(self.seed + 1709 + int(self._fit_count) + int(self._n_seen))
        idx = np.sort(rs.choice(n, size=int(self.tune_max_points), replace=False))
        return X[idx], Y[idx], (None if A is None else A[idx])

    def _select_best_hparams(self):
        X, Y, alpha_subset = self._choose_tuning_subset()
        n = X.shape[0]
        if n <= 2:
            return float(self._best_ell), float(self._best_lam)
        rs = np.random.RandomState(self.seed + 123 + int(self._n_seen) + 31 * int(self._fit_count))
        n_val = int(round(float(self.val_frac) * n))
        n_val = max(1, min(n - 1, n_val))
        perm = rs.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        if tr_idx.size <= 0 or val_idx.size <= 0:
            return float(self._best_ell), float(self._best_lam)
        Xtr = np.asarray(X[tr_idx], dtype=np.float64)
        Ytr = np.asarray(Y[tr_idx], dtype=np.float64)
        Xval = np.asarray(X[val_idx], dtype=np.float64)
        Yval = np.asarray(Y[val_idx], dtype=np.float64)
        use_alpha_weights = (
            str(getattr(self, "_forward_weight_mode", "none")).lower().strip() in ("alpha", "alpha_weighted", "alpha-weighted")
            and alpha_subset is not None
            and np.asarray(alpha_subset).shape[0] == n
        )
        if use_alpha_weights:
            alpha_subset = np.asarray(alpha_subset, dtype=np.float64).reshape(-1)
            w_tr = self._alpha_weights_for(alpha_subset[tr_idx], normalize=True)
            w_val = self._alpha_weights_for(alpha_subset[val_idx], normalize=False)
            if w_tr is None or w_val is None:
                use_alpha_weights = False
        if not use_alpha_weights:
            w_tr = None
            w_val = None
        y_mu, y_sd = standardize_fit(Ytr, sample_weight=w_tr)
        Ys = (Ytr - y_mu) / y_sd
        eye_n = np.eye(Xtr.shape[0], dtype=np.float64)
        best_ell = float(self._best_ell)
        best_lam = float(self._best_lam)
        best_val = float('inf')
        for ell in self.ell_list:
            Ktr = rbf_kernel_np(Xtr, Xtr, float(ell))
            Kval = rbf_kernel_np(Xval, Xtr, float(ell))
            for lam in self.lam_list:
                if w_tr is None:
                    A = Ktr + (float(lam) + float(self.jitter)) * eye_n
                else:
                    A = Ktr + (float(lam) + float(self.jitter)) * np.diag(1.0 / np.maximum(w_tr, 1e-12))
                try:
                    L = np.linalg.cholesky(A)
                    tmp = np.linalg.solve(L, Ys)
                    alpha = np.linalg.solve(L.T, tmp)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.solve(A, Ys)
                pred = (Kval @ alpha) * y_sd + y_mu
                val_rmse = forward_rmse_np(pred, Yval, sample_weight=w_val)
                if val_rmse < best_val:
                    best_val = float(val_rmse)
                    best_ell = float(ell)
                    best_lam = float(lam)
        return best_ell, best_lam

    def _cache_model_to_buffers(self, model_np) -> None:
        device = self.Xtr_buf.device if self.Xtr_buf.numel() else self.emp_mean_buf.device
        with torch.no_grad():
            self.Xtr_buf = torch.from_numpy(model_np['Xtr'].astype(np.float32)).to(device=device, dtype=torch.float32)
            self.alpha_buf = torch.from_numpy(model_np['alpha'].astype(np.float32)).to(device=device, dtype=torch.float32)
            self.y_mu_buf = torch.from_numpy(model_np['y_mu'].astype(np.float32)).to(device=device, dtype=torch.float32)
            self.y_sd_buf = torch.from_numpy(model_np['y_sd'].astype(np.float32)).to(device=device, dtype=torch.float32)
            self.chol_buf = torch.from_numpy(model_np['chol'].astype(np.float32)).to(device=device, dtype=torch.float32)
            self.resid_var_buf = torch.from_numpy(model_np['resid_var'].astype(np.float32)).to(device=device, dtype=torch.float32)

    def _exact_refit(self, do_retune: bool) -> None:
        if self._n_seen <= 0:
            return
        if do_retune:
            self._best_ell, self._best_lam = self._select_best_hparams()
            self._last_tune_n = int(self._n_seen)
        sample_weight = self._current_sample_weights()
        model_np = self._fit_exact_np(
            self._X_hist,
            self._Y_hist,
            ell=float(self._best_ell),
            lam=float(self._best_lam),
            sample_weight=sample_weight,
        )
        self._cache_model_to_buffers(model_np)
        self._n_fit_seen = int(self._n_seen)
        self._fit_count += 1
        self._is_fit = True
        self.eval()

    def partial_fit(self, X_new: np.ndarray, Y_new: np.ndarray, retune: bool = False,
                    sample_alpha_new=None, current_alpha=None,
                    drift_weight_mode: str = "none",
                    alpha_weight_bandwidth: float = 0.15,
                    alpha_weight_floor: float = 1e-4) -> None:
        if X_new is None or Y_new is None:
            return
        X_new = np.asarray(X_new, dtype=np.float32)
        Y_new = np.asarray(Y_new, dtype=np.float32)
        if X_new.ndim == 1:
            X_new = X_new[None, :]
        if Y_new.ndim == 1:
            Y_new = Y_new[None, :]
        if X_new.shape[0] <= 0:
            return
        if X_new.shape[0] != Y_new.shape[0]:
            raise ValueError('X_new and Y_new must have the same number of rows.')
        if sample_alpha_new is None:
            alpha_new = np.full((X_new.shape[0],), np.nan, dtype=np.float64)
        else:
            alpha_new = np.asarray(sample_alpha_new, dtype=np.float64).reshape(-1)
            if alpha_new.size != X_new.shape[0]:
                alpha_new = np.full((X_new.shape[0],), np.nan, dtype=np.float64)
        if self._X_hist.size <= 0:
            self._X_hist = X_new.copy()
            self._Y_hist = Y_new.copy()
            self._alpha_hist = alpha_new.copy()
        else:
            self._X_hist = np.concatenate([self._X_hist, X_new], axis=0)
            self._Y_hist = np.concatenate([self._Y_hist, Y_new], axis=0)
            self._alpha_hist = np.concatenate([self._alpha_hist, alpha_new], axis=0)
        self._n_seen += int(X_new.shape[0])
        self._forward_weight_mode = str(drift_weight_mode)
        self._current_alpha_for_weight = None if current_alpha is None else float(current_alpha)
        self._alpha_weight_bandwidth = float(alpha_weight_bandwidth)
        self._alpha_weight_floor = float(alpha_weight_floor)
        self._update_empirical_stats(Y_new)

        # IMPORTANT POLICY:
        #   - exact/full KRR refit is done on EVERY query update
        #   - hyperparameter sweep (ell, lam) is done ONLY when the caller explicitly
        #     requests retuning (wired to qc checkpoints in the main loop).
        do_retune = bool(retune)
        self._exact_refit(do_retune=do_retune)

    def _fallback_mean_var(self, x: torch.Tensor):
        one = (x.dim() == 1)
        if one:
            x = x.unsqueeze(0)
        n = x.shape[0]
        if self._n_seen > 0:
            mean = self.emp_mean_buf.to(device=x.device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
            var = self.emp_var_buf.to(device=x.device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
        else:
            mean = torch.full((n, self.outcome_dim_est), float(self.default_value), device=x.device, dtype=torch.float32)
            var = torch.full((n, self.outcome_dim_est), 0.05 ** 2, device=x.device, dtype=torch.float32)
        return (mean.squeeze(0), var.squeeze(0)) if one else (mean, var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=torch.float32, device=self.Xtr_buf.device if self.Xtr_buf.numel() else self.emp_mean_buf.device)
        one = (x.dim() == 1)
        if one:
            x = x.unsqueeze(0)
        if (not self._is_fit) or self.Xtr_buf.numel() == 0 or self.alpha_buf.numel() == 0:
            mean, _ = self._fallback_mean_var(x)
            return mean.squeeze(0) if one else mean
        K = rbf_kernel_torch(x, self.Xtr_buf, float(self._best_ell))
        mean_s = K @ self.alpha_buf
        mean = mean_s * self.y_sd_buf.unsqueeze(0) + self.y_mu_buf.unsqueeze(0)
        return mean.squeeze(0) if one else mean

    def predict_mean_var(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.Xtr_buf.device if self.Xtr_buf.numel() else self.emp_mean_buf.device)
        one = (x.dim() == 1)
        if one:
            x = x.unsqueeze(0)
        if (not self._is_fit) or self.Xtr_buf.numel() == 0 or self.alpha_buf.numel() == 0 or self.chol_buf.numel() == 0:
            return self._fallback_mean_var(x)
        K = rbf_kernel_torch(x, self.Xtr_buf, float(self._best_ell))
        mean_s = K @ self.alpha_buf
        mean = mean_s * self.y_sd_buf.unsqueeze(0) + self.y_mu_buf.unsqueeze(0)

        # Exact KRR uncertainty proxy for robust_u_pool.
        # A = K(X,X) + lam I,  v(x) := k(x,X) A^{-1} k(X,x).
        # Near observed points v(x) is close to k(x,x)=1, so epistemic uncertainty should be small;
        # far from the observed prefix v(x) is close to 0, so uncertainty should be large.
        rhs = K.transpose(0, 1).contiguous()
        sol = torch.linalg.solve_triangular(self.chol_buf, rhs, upper=False)
        leverage = torch.sum(sol * sol, dim=0)                  # (n,)
        leverage = torch.clamp(leverage, min=0.0, max=1.0)
        epistemic = torch.clamp(1.0 - leverage, min=0.0, max=1.0)

        scale = torch.maximum(self.y_sd_buf * self.y_sd_buf, self.resid_var_buf)
        var = epistemic[:, None] * scale[None, :]
        var = torch.clamp(var, min=float(self._var_floor))
        return (mean.squeeze(0), var.squeeze(0)) if one else (mean, var)

    def sample_phi(self, X, n_samples: int = 16, rng_np=None):
        if rng_np is None:
            rng_np = np.random
        with torch.no_grad():
            mu_t, var_t = self.predict_mean_var(X)
        mu = mu_t.detach().cpu().numpy().astype(np.float64, copy=False)
        var = var_t.detach().cpu().numpy().astype(np.float64, copy=False)
        if mu.ndim == 1:
            mu = mu[None, :]
            var = var[None, :]
        M = int(max(1, n_samples))
        eps = rng_np.randn(M, mu.shape[0], mu.shape[1]).astype(np.float64)
        samp = mu[None, :, :] + eps * np.sqrt(np.maximum(var[None, :, :], self._var_floor))
        samp = np.clip(samp, 0.0, 1.0)
        return samp.astype(np.float32, copy=False)


def build_forward_model(control_dim: int, outcome_dim_est: int, seed: int = 0) -> nn.Module:
    """Build the Full KRR forward model used in the paper experiments."""
    ell_list, lam_list = _forward_fullkrr_lists_for_control_dim(control_dim)
    return FullKRRForwardModel(
        control_dim=control_dim,
        outcome_dim_est=outcome_dim_est,
        ell_list=ell_list,
        lam_list=lam_list,
        val_frac=float(FORWARD_FULLKRR_VAL_FRAC),
        jitter=float(FORWARD_FULLKRR_JITTER),
        tune_max_points=int(FORWARD_FULLKRR_TUNE_MAX_POINTS),
        warmup_samples=int(FORWARD_FULLKRR_WARMUP_SAMPLES),
        seed=int(seed),
    )



def train_forward_net(net, Xin, Yin, epochs=100, lr=0.01, wd=1e-4, batch_size=256, retune=False,
                      sample_alpha=None, current_alpha=None,
                      drift_weight_mode: str = "none",
                      alpha_weight_bandwidth: float = 0.15,
                      alpha_weight_floor: float = 1e-4):
    """Update the Full KRR forward model with newly observed outcomes."""
    if hasattr(net, "partial_fit"):
        n_seen = int(getattr(net, "n_seen", 0))
        n_total = len(Xin)
        if n_total > n_seen:
            X_new = np.asarray(Xin[n_seen:n_total], dtype=np.float32)
            Y_new = np.asarray(Yin[n_seen:n_total], dtype=np.float32)
            if isinstance(net, FullKRRForwardModel):
                alpha_new = None
                if sample_alpha is not None:
                    alpha_arr = np.asarray(sample_alpha, dtype=np.float64).reshape(-1)
                    if alpha_arr.size >= n_total:
                        alpha_new = alpha_arr[n_seen:n_total]
                net.partial_fit(
                    X_new,
                    Y_new,
                    retune=bool(retune),
                    sample_alpha_new=alpha_new,
                    current_alpha=current_alpha,
                    drift_weight_mode=str(drift_weight_mode),
                    alpha_weight_bandwidth=float(alpha_weight_bandwidth),
                    alpha_weight_floor=float(alpha_weight_floor),
                )
            else:
                net.partial_fit(X_new, Y_new)
        net.eval()
        return
    raise TypeError("This supplementary package expects FullKRRForwardModel.")


def is_forward_tpe_schedule_mode(mode) -> bool:
    m = "" if mode is None else str(mode).lower().strip()
    return m in {
        "tpe_schedule",
        "schedule_tpe",
        "estimated_schedule_tpe",
        "alpha_schedule_tpe",
        "tpe_alpha_schedule",
        "estimated_recency",
        "estimated_recency_weighting",
        "learned_recency",
        "learned_recency_weighting",
    }


def is_forward_oracle_alpha_tpe_mode(mode) -> bool:
    m = "" if mode is None else str(mode).lower().strip()
    return m in {
        "oracle_alpha_tpe",
        "oracle-alpha-tpe",
        "alpha_tpe",
        "alpha-tpe",
        "matched_oracle_alpha_tpe",
        "oracle_alpha_matched_tpe",
        "oracle_recency",
        "oracle_recency_weighting",
        "true_recency",
        "true_recency_weighting",
    }


def estimated_alpha_schedule(q_values, tau_value: float) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    tau = max(float(tau_value), 1e-12)
    return np.clip(1.0 - np.exp(-np.maximum(q_values, 0.0) / tau), 0.0, 1.0)


def _forward_alpha_weights_np(alpha_values,
                              current_alpha: float,
                              h_value: float,
                              *,
                              alpha_weight_floor: float = 1e-4,
                              normalize: bool = True) -> np.ndarray:
    alpha_values = np.asarray(alpha_values, dtype=np.float64).reshape(-1)
    h = max(float(h_value), 1e-12)
    floor = max(float(alpha_weight_floor), 0.0)
    w = np.exp(-0.5 * ((alpha_values - float(current_alpha)) / h) ** 2)
    w = np.maximum(w, floor)
    if normalize:
        wm = float(np.mean(w)) if w.size > 0 else 1.0
        if np.isfinite(wm) and wm > 1e-12:
            w = w / wm
    return np.asarray(w, dtype=np.float64)


def _forward_fullkrr_lists_for_control_dim(control_dim: int):
    if int(control_dim) == 2:
        return list(FORWARD_FULLKRR_ELL_LIST_D2), list(FORWARD_FULLKRR_LAM_LIST_D2)
    return list(FORWARD_FULLKRR_ELL_LIST), list(FORWARD_FULLKRR_LAM_LIST)


def _validation_rmse_weighted_fullkrr_candidate(X: np.ndarray,
                                                Y: np.ndarray,
                                                coord: np.ndarray,
                                                current_coord: float,
                                                h_value: float,
                                                ell: float,
                                                lam: float,
                                                *,
                                                split_seed: int,
                                                alpha_weight_floor: float = 1e-4) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    coord = np.asarray(coord, dtype=np.float64).reshape(-1)
    n = int(X.shape[0])
    if n <= 2:
        return float("inf")
    if coord.shape[0] != n:
        return float("inf")

    rs = np.random.RandomState(int(split_seed))
    n_val = int(round(float(FORWARD_FULLKRR_VAL_FRAC) * n))
    n_val = max(1, min(n - 1, n_val))
    perm = rs.permutation(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    if tr_idx.size <= 0 or val_idx.size <= 0:
        return float("inf")

    Xtr = np.asarray(X[tr_idx], dtype=np.float64)
    Ytr = np.asarray(Y[tr_idx], dtype=np.float64)
    Xval = np.asarray(X[val_idx], dtype=np.float64)
    Yval = np.asarray(Y[val_idx], dtype=np.float64)
    Ctr = np.asarray(coord[tr_idx], dtype=np.float64)
    Cval = np.asarray(coord[val_idx], dtype=np.float64)

    w_tr = _forward_alpha_weights_np(
        Ctr,
        float(current_coord),
        float(h_value),
        alpha_weight_floor=float(alpha_weight_floor),
        normalize=True,
    )
    w_val = _forward_alpha_weights_np(
        Cval,
        float(current_coord),
        float(h_value),
        alpha_weight_floor=float(alpha_weight_floor),
        normalize=False,
    )
    y_mu, y_sd = standardize_fit(Ytr, sample_weight=w_tr)
    Ys = (Ytr - y_mu) / y_sd

    Ktr = rbf_kernel_np(Xtr, Xtr, float(ell))
    A = Ktr + (float(lam) + float(FORWARD_FULLKRR_JITTER)) * np.diag(1.0 / np.maximum(w_tr, 1e-12))
    try:
        L = np.linalg.cholesky(A)
        tmp = np.linalg.solve(L, Ys)
        alpha = np.linalg.solve(L.T, tmp)
    except np.linalg.LinAlgError:
        alpha = np.linalg.solve(A, Ys)
    pred = (rbf_kernel_np(Xval, Xtr, float(ell)) @ alpha) * y_sd + y_mu
    return forward_rmse_np(pred, Yval, sample_weight=w_val)


def select_forward_tpe_schedule_hpo(X_train: np.ndarray,
                                    Y_train: np.ndarray,
                                    q_train: np.ndarray,
                                    q_current: float,
                                    *,
                                    control_dim: int,
                                    seed: int,
                                    split_seed_offset: int = 0,
                                    alpha_weight_floor: float = 1e-4) -> dict:
    import warnings
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"Argument ``multivariate`` is an experimental feature.*",
        category=Warning,
        module=r"optuna\._experimental",
    )

    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    q_train = np.asarray(q_train, dtype=np.float64).reshape(-1)
    n = int(X_train.shape[0])
    ell_list, lam_list = _forward_fullkrr_lists_for_control_dim(control_dim)
    tau_lo, tau_hi = (float(FORWARD_TPE_SCHEDULE_TAU_RANGE[0]), float(FORWARD_TPE_SCHEDULE_TAU_RANGE[1]))
    h_lo, h_hi = (float(FORWARD_TPE_SCHEDULE_H_RANGE[0]), float(FORWARD_TPE_SCHEDULE_H_RANGE[1]))
    ell_lo, ell_hi = (float(min(ell_list)), float(max(ell_list)))
    lam_lo, lam_hi = (float(min(lam_list)), float(max(lam_list)))

    if n <= 2 or q_train.shape[0] != n:
        return dict(
            tau=float(math.sqrt(tau_lo * tau_hi)),
            h=float(math.sqrt(h_lo * h_hi)),
            ell=float(0.5 * (ell_lo + ell_hi)),
            lam=float(math.sqrt(lam_lo * lam_hi)),
            val=float("nan"),
            search_evals=0,
        )

    q_current_int = int(round(float(q_current)))
    hpo_seed = int(seed) + 91001 + int(split_seed_offset) + q_current_int + 101 * int(control_dim)
    sampler = optuna.samplers.TPESampler(
        seed=hpo_seed,
        n_startup_trials=int(FORWARD_TPE_SCHEDULE_STARTUP_TRIALS),
        multivariate=True,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        tau = trial.suggest_float("tau", tau_lo, tau_hi, log=True)
        h = trial.suggest_float("h", h_lo, h_hi, log=True)
        ell = trial.suggest_float("ell", ell_lo, ell_hi, log=True)
        lam = trial.suggest_float("lam", lam_lo, lam_hi, log=True)
        coord = estimated_alpha_schedule(q_train, float(tau))
        current_coord = float(estimated_alpha_schedule(np.asarray([q_current], dtype=np.float64), float(tau))[0])
        try:
            val = _validation_rmse_weighted_fullkrr_candidate(
                X_train,
                Y_train,
                coord,
                current_coord,
                float(h),
                float(ell),
                float(lam),
                split_seed=hpo_seed + 17,
                alpha_weight_floor=float(alpha_weight_floor),
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            val = float("inf")
        if not np.isfinite(float(val)):
            return 1e12
        return float(val)

    study.optimize(
        objective,
        n_trials=int(FORWARD_TPE_SCHEDULE_N_TRIALS),
        show_progress_bar=False,
        catch=(FloatingPointError, ValueError, np.linalg.LinAlgError),
    )

    best_params = dict(study.best_params)
    return dict(
        tau=float(best_params["tau"]),
        h=float(best_params["h"]),
        ell=float(best_params["ell"]),
        lam=float(best_params["lam"]),
        val=float(study.best_value),
        search_evals=int(len(study.trials)),
    )


def train_forward_net_tpe_schedule(net,
                                   Xin,
                                   Yin,
                                   sample_query,
                                   current_query: float,
                                   *,
                                   control_dim: int,
                                   seed: int,
                                   retune: bool = False,
                                   alpha_weight_floor: float = 1e-4):
    """Train FullKRR with a TPE-selected estimated alpha schedule.

    TPE is run at the same moments as FullKRR retuning.  Between retunes, the
    previously selected tau/h/ell/lambda are reused while current alpha advances.
    """
    if not isinstance(net, FullKRRForwardModel):
        raise ValueError("forward_weight_mode='tpe_schedule' requires FullKRRForwardModel.")

    X = np.asarray(Xin, dtype=np.float32)
    Y = np.asarray(Yin, dtype=np.float32)
    q = np.asarray(sample_query, dtype=np.float64).reshape(-1)
    n_total = int(X.shape[0])
    if n_total <= 0:
        net.eval()
        return
    if q.shape[0] != n_total:
        raise ValueError("sample_query must have one entry per forward-training sample.")

    previous = getattr(net, "_tpe_schedule_params", None)
    need_select = bool(retune) or previous is None
    if need_select:
        best = select_forward_tpe_schedule_hpo(
            X,
            Y,
            q,
            float(current_query),
            control_dim=int(control_dim),
            seed=int(seed),
            alpha_weight_floor=float(alpha_weight_floor),
        )
        net._tpe_schedule_params = dict(best)
    else:
        best = dict(previous)

    coord = estimated_alpha_schedule(q, float(best["tau"]))
    current_coord = float(estimated_alpha_schedule(np.asarray([current_query], dtype=np.float64), float(best["tau"]))[0])

    n_seen = int(getattr(net, "n_seen", 0))
    had_new_samples = n_total > n_seen
    if n_seen > 0:
        n_replace = min(n_seen, n_total)
        if net._alpha_hist.shape[0] == n_seen:
            net._alpha_hist[:n_replace] = coord[:n_replace]

    net._best_ell = float(best["ell"])
    net._best_lam = float(best["lam"])
    net._forward_weight_mode = "alpha"
    net._current_alpha_for_weight = float(current_coord)
    net._alpha_weight_bandwidth = float(best["h"])
    net._alpha_weight_floor = float(alpha_weight_floor)
    net._last_tpe_schedule_q = float(current_query)
    net._last_tpe_schedule_val = float(best.get("val", np.nan))
    net._last_tpe_schedule_search_evals = int(best.get("search_evals", 0))

    train_forward_net(
        net,
        X,
        Y,
        retune=False,
        sample_alpha=coord,
        current_alpha=float(current_coord),
        drift_weight_mode="alpha",
        alpha_weight_bandwidth=float(best["h"]),
        alpha_weight_floor=float(alpha_weight_floor),
    )

    if (not had_new_samples) and int(getattr(net, "n_seen", 0)) == n_total and net._alpha_hist.shape[0] == n_total:
        # A retuned tau changes the coordinates of older samples; ensure the
        # cached exact fit uses the full new schedule even if no samples were appended.
        net._alpha_hist = coord.astype(np.float64, copy=True)
        net._exact_refit(do_retune=False)
    net.eval()


def select_forward_tpe_oracle_alpha_hpo(X_train: np.ndarray,
                                        Y_train: np.ndarray,
                                        sample_alpha: np.ndarray,
                                        current_alpha: float,
                                        *,
                                        control_dim: int,
                                        seed: int,
                                        split_seed_offset: int = 0,
                                        alpha_weight_floor: float = 1e-4) -> dict:
    import warnings
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"Argument ``multivariate`` is an experimental feature.*",
        category=Warning,
        module=r"optuna\._experimental",
    )

    X_train = np.asarray(X_train, dtype=np.float64)
    Y_train = np.asarray(Y_train, dtype=np.float64)
    coord = np.asarray(sample_alpha, dtype=np.float64).reshape(-1)
    n = int(X_train.shape[0])
    ell_list, lam_list = _forward_fullkrr_lists_for_control_dim(control_dim)
    h_lo, h_hi = (float(FORWARD_TPE_SCHEDULE_H_RANGE[0]), float(FORWARD_TPE_SCHEDULE_H_RANGE[1]))
    ell_lo, ell_hi = (float(min(ell_list)), float(max(ell_list)))
    lam_lo, lam_hi = (float(min(lam_list)), float(max(lam_list)))

    if n <= 2 or coord.shape[0] != n or not np.all(np.isfinite(coord)) or not np.isfinite(float(current_alpha)):
        return dict(
            h=float(math.sqrt(h_lo * h_hi)),
            ell=float(0.5 * (ell_lo + ell_hi)),
            lam=float(math.sqrt(lam_lo * lam_hi)),
            val=float("nan"),
            search_evals=0,
        )

    current_alpha_int = int(round(1000.0 * float(current_alpha)))
    hpo_seed = int(seed) + 92001 + int(split_seed_offset) + current_alpha_int + 101 * int(control_dim)
    sampler = optuna.samplers.TPESampler(
        seed=hpo_seed,
        n_startup_trials=int(FORWARD_TPE_SCHEDULE_STARTUP_TRIALS),
        multivariate=True,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        h = trial.suggest_float("h", h_lo, h_hi, log=True)
        ell = trial.suggest_float("ell", ell_lo, ell_hi, log=True)
        lam = trial.suggest_float("lam", lam_lo, lam_hi, log=True)
        try:
            val = _validation_rmse_weighted_fullkrr_candidate(
                X_train,
                Y_train,
                coord,
                float(current_alpha),
                float(h),
                float(ell),
                float(lam),
                split_seed=hpo_seed + 17,
                alpha_weight_floor=float(alpha_weight_floor),
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            val = float("inf")
        if not np.isfinite(float(val)):
            return 1e12
        return float(val)

    study.optimize(
        objective,
        n_trials=int(FORWARD_TPE_SCHEDULE_N_TRIALS),
        show_progress_bar=False,
        catch=(FloatingPointError, ValueError, np.linalg.LinAlgError),
    )

    best_params = dict(study.best_params)
    return dict(
        h=float(best_params["h"]),
        ell=float(best_params["ell"]),
        lam=float(best_params["lam"]),
        val=float(study.best_value),
        search_evals=int(len(study.trials)),
    )


def train_forward_net_oracle_alpha_tpe(net,
                                       Xin,
                                       Yin,
                                       sample_alpha,
                                       current_alpha: float,
                                       *,
                                       control_dim: int,
                                       seed: int,
                                       retune: bool = False,
                                       alpha_weight_floor: float = 1e-4):
    """Train FullKRR with true alpha coordinates and TPE-selected h/ell/lambda."""
    if not isinstance(net, FullKRRForwardModel):
        raise ValueError("forward_weight_mode='oracle_alpha_tpe' requires FullKRRForwardModel.")

    X = np.asarray(Xin, dtype=np.float32)
    Y = np.asarray(Yin, dtype=np.float32)
    coord = np.asarray(sample_alpha, dtype=np.float64).reshape(-1)
    n_total = int(X.shape[0])
    if n_total <= 0:
        net.eval()
        return
    if coord.shape[0] != n_total:
        raise ValueError("sample_alpha must have one entry per forward-training sample.")

    previous = getattr(net, "_oracle_alpha_tpe_params", None)
    need_select = bool(retune) or previous is None
    if need_select:
        best = select_forward_tpe_oracle_alpha_hpo(
            X,
            Y,
            coord,
            float(current_alpha),
            control_dim=int(control_dim),
            seed=int(seed),
            alpha_weight_floor=float(alpha_weight_floor),
        )
        net._oracle_alpha_tpe_params = dict(best)
    else:
        best = dict(previous)

    n_seen = int(getattr(net, "n_seen", 0))
    had_new_samples = n_total > n_seen
    if n_seen > 0:
        n_replace = min(n_seen, n_total)
        if net._alpha_hist.shape[0] == n_seen:
            net._alpha_hist[:n_replace] = coord[:n_replace]

    net._best_ell = float(best["ell"])
    net._best_lam = float(best["lam"])
    net._forward_weight_mode = "alpha"
    net._current_alpha_for_weight = float(current_alpha)
    net._alpha_weight_bandwidth = float(best["h"])
    net._alpha_weight_floor = float(alpha_weight_floor)
    net._last_oracle_alpha_tpe_val = float(best.get("val", np.nan))
    net._last_oracle_alpha_tpe_search_evals = int(best.get("search_evals", 0))

    train_forward_net(
        net,
        X,
        Y,
        retune=False,
        sample_alpha=coord,
        current_alpha=float(current_alpha),
        drift_weight_mode="alpha",
        alpha_weight_bandwidth=float(best["h"]),
        alpha_weight_floor=float(alpha_weight_floor),
    )

    if (not had_new_samples) and int(getattr(net, "n_seen", 0)) == n_total and net._alpha_hist.shape[0] == n_total:
        net._alpha_hist = coord.astype(np.float64, copy=True)
        net._exact_refit(do_retune=False)
    net.eval()


def epochs_for_forward(control_dim: int) -> int:
    return 20 + 4 * control_dim

def compute_candidate_pool_size(control_dim: int) -> int:
    return max(4000, 400 * int(control_dim))


def bern_entropy_np(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def parse_joint_mode_spec(mode: str):
    m = "" if mode is None else str(mode).lower().strip()

    joint_soft_l0 = {
        'joint_soft', 'joint_soft_lam0', 'joint_soft_lam_0', 'joint_soft_l0', 'joint_soft_0'
    }
    joint_soft_l05 = {
        'joint_soft_lam05', 'joint_soft_lam0.5', 'joint_soft_lam_0.5',
        'joint_soft_l05', 'joint_soft_05', 'joint_soft_fwd'
    }
    joint_mi_l0 = {
        'joint_mi', 'joint_mi_lam0', 'joint_mi_lam_0', 'joint_mi_l0', 'joint_mi_0'
    }
    joint_mi_l05 = {
        'joint_mi_lam05', 'joint_mi_lam0.5', 'joint_mi_lam_0.5',
        'joint_mi_l05', 'joint_mi_05', 'joint_mi_fwd'
    }

    if m in joint_soft_l0:
        return 'joint_soft_lam0', 'soft', 0.0
    if m in joint_soft_l05:
        return 'joint_soft_lam05', 'soft', float(JOINT_LAMBDA_HALF)
    if m in joint_mi_l0:
        return 'joint_mi_lam0', 'mi', 0.0
    if m in joint_mi_l05:
        return 'joint_mi_lam05', 'mi', float(JOINT_LAMBDA_HALF)
    return None, None, None


def parse_boundary_lookahead_aux_mode(mode: str):
    m = "" if mode is None else str(mode).lower().strip()
    cond_aliases = {
        'cond_soft', 'conditioned_soft', 'conditioned_soft_pool', 'cond_soft_pool', 'soft_cond'
    }
    boundary_lookahead_aliases = {
        'boundary_lookahead_soft'
    }
    if m in cond_aliases:
        return 'cond_soft'
    if m in boundary_lookahead_aliases:
        return 'boundary_lookahead_soft'
    return None


def make_empty_joint_sel_debug(mode=None):
    return dict(
        mode=(None if mode is None else str(mode)),
        score=np.nan,
        main=np.nan,
        bonus=np.nan,
        multiplier=np.nan,
        pbar=np.nan,
        aux_fwd=np.nan,
        aux_inv=np.nan,
        target_cur=np.nan,
        target_post=np.nan,
        target_gain=np.nan,
        reward_gain=np.nan,
        forward_reduction=np.nan,
        target_gap=np.nan,
        shortlist_src_softopt=np.nan,
        shortlist_src_pool_soft=np.nan,
        shortlist_src_pool_var=np.nan,
        shortlist_src_random=np.nan,
    )


def make_empty_gp_sel_debug(mode=None):
    out = dict(mode=(None if mode is None else str(mode)))
    for _, key in GP_SELDIAG_SPECS:
        out[key] = np.nan
    return out


def gp_sel_debug_to_array(sel_dbg):
    sel_dbg = {} if sel_dbg is None else dict(sel_dbg)
    return np.asarray(
        [float(sel_dbg.get(key, np.nan)) for _, key in GP_SELDIAG_SPECS],
        dtype=np.float64,
    )


def compute_joint_anchor_scores_from_candidates(
    X_cand,
    phi_cand_mean,
    phi_anchor,
    forward_net,
    theta_sel,
    cfg,
    *,
    joint_kind: str,
    lam: float,
    use_oracle_phi_for_U: bool,
    rng_np=None,
    n_fwd_samples: int = JOINT_FWD_SAMPLES,
    beta_vec=None,
    beta_scalar: float = 1.0,
):
    if rng_np is None:
        rng_np = np.random

    X_cand = np.asarray(X_cand, dtype=np.float32)
    if X_cand.ndim == 1:
        X_cand = X_cand[None, :]

    phi_cand_mean = np.asarray(phi_cand_mean, dtype=np.float64)
    if phi_cand_mean.ndim == 1:
        phi_cand_mean = phi_cand_mean[None, :]

    phi_anchor = np.asarray(phi_anchor, dtype=np.float64).reshape(1, -1)
    theta_sel = np.asarray(theta_sel, dtype=np.float64)

    M_req = int(max(1, n_fwd_samples))

    if use_oracle_phi_for_U:
        Phi_samp = np.repeat(phi_cand_mean[None, :, :], M_req, axis=0)
    else:
        if hasattr(forward_net, 'sample_phi') and int(getattr(forward_net, 'n_seen', 0)) > 0:
            Phi_samp = forward_net.sample_phi(X_cand, n_samples=M_req, rng_np=rng_np)
            Phi_samp = np.asarray(Phi_samp, dtype=np.float64)
        else:
            Phi_samp = np.repeat(phi_cand_mean[None, :, :], M_req, axis=0)

    if Phi_samp.ndim == 2:
        Phi_samp = Phi_samp[None, :, :]

    M_eff, N_eval, _ = Phi_samp.shape
    r_anchor = nn_reward_vectorized(theta_sel, phi_anchor, cfg).reshape(-1).astype(np.float64, copy=False)

    pbar_m = np.empty((M_eff, N_eval), dtype=np.float64)
    if joint_kind == 'soft':
        aux_inv_acc = np.zeros((N_eval,), dtype=np.float64)
    elif joint_kind == 'mi':
        H_ps_acc = np.zeros((N_eval,), dtype=np.float64)
    else:
        raise ValueError(f"joint_kind must be 'soft' or 'mi', got {joint_kind!r}")

    for m in range(M_eff):
        phi_m = np.asarray(Phi_samp[m], dtype=np.float64)
        r_cand_m = nn_reward_vectorized(theta_sel, phi_m, cfg).astype(np.float64, copy=False)
        diff_m = r_anchor[:, None] - r_cand_m

        if beta_vec is None:
            logits_m = float(beta_scalar) * diff_m
        else:
            logits_m = beta_vec * diff_m

        p_s_m = _sigmoid(logits_m)
        pbar_m[m, :] = np.mean(p_s_m, axis=0)

        if joint_kind == 'soft':
            aux_inv_acc += np.var(p_s_m, axis=0)
        else:
            H_ps_acc += np.mean(bern_entropy_np(p_s_m), axis=0)

    pbar = np.mean(pbar_m, axis=0)

    if joint_kind == 'soft':
        main = 4.0 * pbar * (1.0 - pbar)
        aux_fwd = np.var(pbar_m, axis=0)
        bonus = np.clip(4.0 * aux_fwd, 0.0, 1.0)
        aux_inv = aux_inv_acc / float(max(1, M_eff))
    else:
        H_bar = bern_entropy_np(pbar)
        H_bar_m = np.mean(bern_entropy_np(pbar_m), axis=0)
        E_H = H_ps_acc / float(max(1, M_eff))
        main = np.maximum(H_bar - E_H, 0.0)
        aux_fwd = np.maximum(H_bar - H_bar_m, 0.0)                # I(Y; M)
        bonus = np.clip(aux_fwd / np.log(2.0), 0.0, 1.0)          # normalized to [0,1]
        aux_inv = np.maximum(H_bar_m - E_H, 0.0)                  # I(Y; S | M)

    multiplier = 1.0 + float(lam) * bonus
    score = main * multiplier

    return dict(
        score=np.asarray(score, dtype=np.float64),
        main=np.asarray(main, dtype=np.float64),
        fwd_bonus=np.asarray(bonus, dtype=np.float64),
        multiplier=np.asarray(multiplier, dtype=np.float64),
        pbar=np.asarray(pbar, dtype=np.float64),
        aux_fwd=np.asarray(aux_fwd, dtype=np.float64),
        aux_inv=np.asarray(aux_inv, dtype=np.float64),
    )


def _module_device(net):
    try:
        return next(net.parameters()).device
    except StopIteration:
        try:
            return next(net.buffers()).device
        except StopIteration:
            return torch.device('cpu')


def get_forward_mean_var_np(forward_net, X):
    X = np.asarray(X, dtype=np.float32)
    one = (X.ndim == 1)
    if one:
        X = X[None, :]
    dev = _module_device(forward_net)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        if hasattr(forward_net, 'predict_mean_var'):
            mean_t, var_t = forward_net.predict_mean_var(Xt)
        else:
            mean_t = forward_net(Xt)
            var_t = torch.full_like(mean_t, 1e-4)
    mean = mean_t.detach().cpu().numpy().astype(np.float64, copy=False)
    var = np.maximum(var_t.detach().cpu().numpy().astype(np.float64, copy=False), 1e-6)
    if one:
        return mean[0], var[0]
    return mean, var


def _forward_residual_var_vec(forward_net, d_out: int) -> np.ndarray:
    if hasattr(forward_net, 'resid_var_buf'):
        try:
            rv = forward_net.resid_var_buf.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
            if rv.size == int(d_out):
                return np.maximum(rv, 0.0)
        except Exception:
            pass
    return np.zeros((int(d_out),), dtype=np.float64)


def get_forward_mean_var_decomposed_np(forward_net, X, noise_cfg=None):
    """Split predictive uncertainty into reducible epistemic and irreducible observation noise."""
    X = np.asarray(X, dtype=np.float32)
    one = (X.ndim == 1)
    if one:
        X = X[None, :]

    mean_np, var_np = get_forward_mean_var_np(forward_net, X)
    mean_np = np.asarray(mean_np, dtype=np.float64)
    var_np = np.asarray(var_np, dtype=np.float64)
    if mean_np.ndim == 1:
        mean_np = mean_np[None, :]
        var_np = var_np[None, :]

    d_out = int(mean_np.shape[1])
    resid_vec = _forward_residual_var_vec(forward_net, d_out)
    phi_noise_var = 0.0 if noise_cfg is None else float(getattr(noise_cfg, 'oracle_phi_noise_std', 0.0)) ** 2
    obs_floor = np.maximum(resid_vec, 0.0) + max(float(phi_noise_var), 0.0)
    var_obs = np.broadcast_to(obs_floor[None, :], mean_np.shape).astype(np.float64, copy=True)
    var_epi = np.maximum(var_np, 1e-6)
    epi_cur = None
    scale_epi = None
    can_update = False

    if (
        isinstance(forward_net, FullKRRForwardModel)
        and bool(getattr(forward_net, '_is_fit', False))
        and forward_net.Xtr_buf.numel() > 0
        and forward_net.chol_buf.numel() > 0
    ):
        dev = _module_device(forward_net)
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        with torch.no_grad():
            K = rbf_kernel_torch(Xt, forward_net.Xtr_buf, float(getattr(forward_net, '_best_ell', 0.65)))
            rhs = K.transpose(0, 1).contiguous()
            sol = torch.linalg.solve_triangular(forward_net.chol_buf, rhs, upper=False)
            leverage = torch.sum(sol * sol, dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
        leverage = np.clip(leverage, 0.0, 1.0)
        epi_cur = np.clip(1.0 - leverage, 0.0, 1.0)
        y_var = forward_net.y_sd_buf.detach().cpu().numpy().astype(np.float64, copy=False) ** 2
        scale_epi = np.maximum(y_var - resid_vec, 1e-6)
        var_epi = np.maximum(epi_cur[:, None] * scale_epi[None, :], 1e-6)
        can_update = True

    var_total = np.maximum(var_epi + var_obs, 1e-6)
    out = dict(
        mean=np.asarray(mean_np, dtype=np.float64),
        var_total=np.asarray(var_total, dtype=np.float64),
        var_epi=np.asarray(var_epi, dtype=np.float64),
        var_obs=np.asarray(var_obs, dtype=np.float64),
        epi_cur=(None if epi_cur is None else np.asarray(epi_cur, dtype=np.float64)),
        scale_epi=(None if scale_epi is None else np.asarray(scale_epi, dtype=np.float64)),
        can_update=bool(can_update),
    )

    if one:
        return dict(
            mean=out['mean'][0],
            var_total=out['var_total'][0],
            var_epi=out['var_epi'][0],
            var_obs=out['var_obs'][0],
            epi_cur=out['epi_cur'],
            scale_epi=out['scale_epi'],
            can_update=out['can_update'],
        )
    return out


def sample_phi_from_mean_var_np(phi_mean, phi_var, n_samples: int, rng_np=None):
    if rng_np is None:
        rng_np = np.random
    mu = np.asarray(phi_mean, dtype=np.float64)
    var = np.maximum(np.asarray(phi_var, dtype=np.float64), 1e-8)
    if mu.ndim == 1:
        mu = mu[None, :]
        var = var[None, :]
    M = int(max(1, n_samples))
    eps = rng_np.randn(M, mu.shape[0], mu.shape[1]).astype(np.float64)
    samp = mu[None, :, :] + eps * np.sqrt(var[None, :, :])
    samp = np.clip(samp, 0.0, 1.0)
    return samp.astype(np.float32, copy=False)


def compute_anchor_soft_scores_from_mean_phi(
    phi_cand,
    phi_anchor,
    theta_sel,
    cfg,
    *,
    beta_vec=None,
    beta_scalar: float = 1.0,
):
    phi_cand = np.asarray(phi_cand, dtype=np.float64)
    if phi_cand.ndim == 1:
        phi_cand = phi_cand[None, :]
    phi_anchor = np.asarray(phi_anchor, dtype=np.float64).reshape(1, -1)
    theta_sel = np.asarray(theta_sel, dtype=np.float64)
    r_anchor = nn_reward_vectorized(theta_sel, phi_anchor, cfg).reshape(-1).astype(np.float64, copy=False)
    r_cand = nn_reward_vectorized(theta_sel, phi_cand, cfg).astype(np.float64, copy=False)
    diff = r_anchor[:, None] - r_cand
    if beta_vec is None:
        logits = float(beta_scalar) * diff
    else:
        logits = beta_vec * diff
    p_s = _sigmoid(logits)
    pbar = np.mean(p_s, axis=0)
    U = 4.0 * pbar * (1.0 - pbar)
    return dict(score=U.astype(np.float64), pbar=pbar.astype(np.float64), p_s=p_s.astype(np.float64, copy=False))


def compute_cond_soft_anchor_scores_from_candidates(
    X_cand,
    phi_cand_mean,
    phi_anchor,
    forward_net,
    theta_sel,
    cfg,
    *,
    use_oracle_phi_for_U: bool,
    rng_np=None,
    n_fwd_samples: int = CONDITIONED_SOFT_FORWARD_SAMPLES,
    beta_vec=None,
    beta_scalar: float = 1.0,
):
    if rng_np is None:
        rng_np = np.random

    X_cand = np.asarray(X_cand, dtype=np.float32)
    if X_cand.ndim == 1:
        X_cand = X_cand[None, :]
    phi_cand_mean = np.asarray(phi_cand_mean, dtype=np.float64)
    if phi_cand_mean.ndim == 1:
        phi_cand_mean = phi_cand_mean[None, :]
    phi_anchor = np.asarray(phi_anchor, dtype=np.float64).reshape(1, -1)
    theta_sel = np.asarray(theta_sel, dtype=np.float64)
    M_req = int(max(1, n_fwd_samples))

    if use_oracle_phi_for_U:
        Phi_samp = np.repeat(phi_cand_mean[None, :, :], M_req, axis=0)
    else:
        if hasattr(forward_net, 'sample_phi') and int(getattr(forward_net, 'n_seen', 0)) > 0:
            Phi_samp = np.asarray(forward_net.sample_phi(X_cand, n_samples=M_req, rng_np=rng_np), dtype=np.float64)
        else:
            Phi_samp = np.repeat(phi_cand_mean[None, :, :], M_req, axis=0)
    if Phi_samp.ndim == 2:
        Phi_samp = Phi_samp[None, :, :]

    M_eff, N_eval, _ = Phi_samp.shape
    r_anchor = nn_reward_vectorized(theta_sel, phi_anchor, cfg).reshape(-1).astype(np.float64, copy=False)
    pbar_m = np.empty((M_eff, N_eval), dtype=np.float64)
    aux_inv_acc = np.zeros((N_eval,), dtype=np.float64)

    for m in range(M_eff):
        phi_m = np.asarray(Phi_samp[m], dtype=np.float64)
        r_cand_m = nn_reward_vectorized(theta_sel, phi_m, cfg).astype(np.float64, copy=False)
        diff_m = r_anchor[:, None] - r_cand_m
        if beta_vec is None:
            logits_m = float(beta_scalar) * diff_m
        else:
            logits_m = beta_vec * diff_m
        p_s_m = _sigmoid(logits_m)
        q_m = np.mean(p_s_m, axis=0)
        pbar_m[m, :] = q_m
        aux_inv_acc += np.var(p_s_m, axis=0)

    main = np.mean(4.0 * pbar_m * (1.0 - pbar_m), axis=0)
    pbar = np.mean(pbar_m, axis=0)
    aux_fwd = np.var(pbar_m, axis=0)
    aux_inv = aux_inv_acc / float(max(1, M_eff))
    return dict(
        score=np.asarray(main, dtype=np.float64),
        main=np.asarray(main, dtype=np.float64),
        fwd_bonus=np.zeros_like(main, dtype=np.float64),
        multiplier=np.ones_like(main, dtype=np.float64),
        pbar=np.asarray(pbar, dtype=np.float64),
        aux_fwd=np.asarray(aux_fwd, dtype=np.float64),
        aux_inv=np.asarray(aux_inv, dtype=np.float64),
    )


def compute_predictive_pbar_from_phi_array(
    theta_sel,
    phi_anchor,
    phi_targets,
    cfg,
    *,
    beta_vec=None,
    beta_scalar: float = 1.0,
    reward_weights=None,
):
    phi_targets = np.asarray(phi_targets, dtype=np.float64)
    if phi_targets.ndim == 1:
        phi_targets = phi_targets[None, :]
    phi_anchor = np.asarray(phi_anchor, dtype=np.float64).reshape(1, -1)
    theta_sel = np.asarray(theta_sel, dtype=np.float64)
    r_anchor = nn_reward_vectorized(theta_sel, phi_anchor, cfg).reshape(-1).astype(np.float64, copy=False)
    r_t = nn_reward_vectorized(theta_sel, phi_targets, cfg).astype(np.float64, copy=False)
    diff = r_anchor[:, None] - r_t
    if beta_vec is None:
        logits = float(beta_scalar) * diff
    else:
        logits = beta_vec * diff
    p_s = _sigmoid(logits)

    if reward_weights is None:
        pbar = np.mean(p_s, axis=0)
    else:
        w = np.asarray(reward_weights, dtype=np.float64).reshape(-1, 1)
        wsum = float(np.sum(w))
        if not np.isfinite(wsum) or wsum <= 1e-12:
            pbar = np.mean(p_s, axis=0)
        else:
            pbar = np.sum(w * p_s, axis=0) / wsum
    return np.asarray(pbar, dtype=np.float64)


def conditional_soft_uncertainty_from_phi_samples(
    theta_sel,
    phi_anchor,
    Phi_samp,
    cfg,
    *,
    beta_vec=None,
    beta_scalar: float = 1.0,
    reward_weights=None,
):
    Phi_samp = np.asarray(Phi_samp, dtype=np.float64)
    if Phi_samp.ndim == 2:
        Phi_samp = Phi_samp[None, :, :]
    acc = 0.0
    for m in range(Phi_samp.shape[0]):
        q = compute_predictive_pbar_from_phi_array(
            theta_sel, phi_anchor, Phi_samp[m], cfg,
            beta_vec=beta_vec, beta_scalar=beta_scalar,
            reward_weights=reward_weights,
        )
        acc += float(np.mean(4.0 * q * (1.0 - q)))
    return float(acc / float(max(1, Phi_samp.shape[0])))


def build_target_set_soft_pool(
    X_cand,
    phi_cand_mean,
    phi_anchor,
    theta_sel,
    cfg,
    *,
    topk: int = BOUNDARY_LOOKAHEAD_SOFT_TARGET_TOPK,
    beta_vec=None,
    beta_scalar: float = 1.0,
):
    stats = compute_anchor_soft_scores_from_mean_phi(
        phi_cand_mean, phi_anchor, theta_sel, cfg,
        beta_vec=beta_vec, beta_scalar=beta_scalar,
    )
    score = np.asarray(stats['score'], dtype=np.float64)
    topk = int(max(1, min(int(topk), int(score.shape[0]))))
    idx = np.argsort(-score)[:topk]
    return (
        np.asarray(X_cand[idx], dtype=np.float32),
        np.asarray(phi_cand_mean[idx], dtype=np.float64),
        np.asarray(score[idx], dtype=np.float64),
        np.asarray(stats['pbar'], dtype=np.float64)[idx],
    )


def prepare_target_forward_state(forward_net, X_target, noise_cfg=None):
    X_target = np.asarray(X_target, dtype=np.float32)
    pack = get_forward_mean_var_decomposed_np(forward_net, X_target, noise_cfg=noise_cfg)
    mean_t = np.asarray(pack['mean'], dtype=np.float64)
    var_total = np.asarray(pack['var_total'], dtype=np.float64)
    var_epi = np.asarray(pack['var_epi'], dtype=np.float64)
    var_obs = np.asarray(pack['var_obs'], dtype=np.float64)
    if mean_t.ndim == 1:
        mean_t = mean_t[None, :]
        var_total = var_total[None, :]
        var_epi = var_epi[None, :]
        var_obs = var_obs[None, :]

    state = dict(
        X_target=np.asarray(X_target, dtype=np.float64),
        mean=np.asarray(mean_t, dtype=np.float64),
        var=np.maximum(np.asarray(var_total, dtype=np.float64), 1e-6),
        var_epi=np.maximum(np.asarray(var_epi, dtype=np.float64), 1e-6),
        var_obs=np.maximum(np.asarray(var_obs, dtype=np.float64), 0.0),
        scale=None,
        epi_cur=None,
        can_update=False,
        Xtr=None,
        chol=None,
        ell=None,
        lam_obs=None,
        K_TX=None,
        Ainv_KTX_T=None,
    )

    if isinstance(forward_net, FullKRRForwardModel) and bool(getattr(forward_net, '_is_fit', False)) and forward_net.Xtr_buf.numel() > 0 and forward_net.chol_buf.numel() > 0:
        Xtr = forward_net.Xtr_buf.detach().cpu().numpy().astype(np.float64, copy=False)
        chol = forward_net.chol_buf.detach().cpu().numpy().astype(np.float64, copy=False)
        ell = float(getattr(forward_net, '_best_ell', 0.65))
        lam_obs = float(getattr(forward_net, '_best_lam', 1e-2)) + float(getattr(forward_net, 'jitter', 0.0))
        K_TX = rbf_kernel_np(X_target.astype(np.float64), Xtr, ell)
        epi_cur = np.asarray(pack.get('epi_cur', np.zeros((mean_t.shape[0],), dtype=np.float64)), dtype=np.float64).reshape(-1)
        scale = np.asarray(pack.get('scale_epi', np.mean(var_epi, axis=0)), dtype=np.float64).reshape(-1)
        state.update(
            scale=np.asarray(scale, dtype=np.float64),
            epi_cur=np.asarray(epi_cur, dtype=np.float64),
            can_update=True,
            Xtr=np.asarray(Xtr, dtype=np.float64),
            chol=np.asarray(chol, dtype=np.float64),
            ell=float(ell),
            lam_obs=float(max(lam_obs, 1e-8)),
            K_TX=np.asarray(K_TX, dtype=np.float64),
        )
    else:
        var_safe = np.maximum(var_epi, 1e-8)
        scale = np.maximum(np.mean(var_safe, axis=0), 1e-6)
        epi_cur = np.clip(np.mean(var_safe / scale[None, :], axis=1), 0.0, 1.0)
        state.update(
            scale=np.asarray(scale, dtype=np.float64),
            epi_cur=np.asarray(epi_cur, dtype=np.float64),
            can_update=False,
        )
    return state


def updated_target_epistemic_from_state(target_state, x_new):
    epi_cur = np.asarray(target_state['epi_cur'], dtype=np.float64)
    if (not bool(target_state.get('can_update', False))) or epi_cur.size <= 0:
        return epi_cur.copy()

    x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
    ell = float(target_state['ell'])
    Xtr = np.asarray(target_state['Xtr'], dtype=np.float64)
    chol = np.asarray(target_state['chol'], dtype=np.float64)
    K_TX = np.asarray(target_state['K_TX'], dtype=np.float64)
    lam_obs = float(target_state['lam_obs'])

    k_xX = rbf_kernel_np(x_new, Xtr, ell).reshape(-1)                 # (n,)
    v_x = np.linalg.solve(chol, k_xX.reshape(-1, 1)).reshape(-1)      # (n,)
    epi_x = float(np.clip(1.0 - np.sum(v_x * v_x), 0.0, 1.0))
    a_x = np.linalg.solve(chol.T, v_x.reshape(-1, 1)).reshape(-1)     # A^{-1} k_x

    k_tx = rbf_kernel_np(np.asarray(target_state['X_target'], dtype=np.float64), x_new, ell).reshape(-1)
    cross = k_tx - K_TX @ a_x
    denom = max(epi_x + lam_obs, 1e-8)
    epi_new = epi_cur - (cross * cross) / denom
    epi_new = np.clip(epi_new, 0.0, epi_cur)
    return np.asarray(epi_new, dtype=np.float64)


def build_boundary_lookahead_shortlist(
    X_cand,
    phi_cand_mean,
    phi_anchor,
    theta_sel,
    cfg,
    *,
    control_dim: int,
    forward_net,
    outcome_model,
    est_indices,
    use_oracle_phi_for_U: bool,
    beta_samples_np=None,
    beta_vec=None,
    beta_scalar: float = 1.0,
    device=None,
    rng_np=None,
):
    if rng_np is None:
        rng_np = np.random
    quota_softopt = 0
    quota_pool_soft = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_KEEP + BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_SOFT_EXTRA))
    quota_pool_var = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_VARIANCE_EXTRA))
    quota_pool_random = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_RANDOM_EXTRA))
    target_total = int(max(1, quota_softopt + quota_pool_soft + quota_pool_var + quota_pool_random))

    def _key_from_x(x):
        return tuple(np.round(np.asarray(x, dtype=np.float64), 6))

    source_lists = dict(
        softopt=[],
        pool_soft=[],
        pool_var=[],
        pool_random=[],
    )

    # Pool-only shortlist: skip soft-optimization proposals and spend the budget on ranked pool picks.

    mean_soft = compute_anchor_soft_scores_from_mean_phi(
        phi_cand_mean, phi_anchor, theta_sel, cfg,
        beta_vec=beta_vec, beta_scalar=beta_scalar,
    )
    U_pool = np.asarray(mean_soft['score'], dtype=np.float64)
    ord_soft = np.argsort(-np.nan_to_num(U_pool, nan=-1e18))
    source_lists['pool_soft'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in ord_soft]

    try:
        _, var_pool = get_forward_mean_var_np(forward_net, X_cand)
        var_score = np.mean(np.asarray(var_pool, dtype=np.float64), axis=1)
        ord_var = np.argsort(-np.nan_to_num(var_score, nan=-1e18))
        source_lists['pool_var'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in ord_var]
    except Exception:
        pass

    if int(X_cand.shape[0]) > 0:
        perm_rand = np.asarray(rng_np.permutation(int(X_cand.shape[0])), dtype=np.int64).reshape(-1)
        source_lists['pool_random'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in perm_rand]

    if all(len(v) <= 0 for v in source_lists.values()):
        best_idx = int(np.nanargmax(U_pool)) if np.isfinite(U_pool).any() else int(rng_np.randint(0, int(X_cand.shape[0])))
        x_best = np.asarray(X_cand[best_idx:best_idx+1], dtype=np.float32)
        return dict(
            X=x_best,
            src_softopt=np.array([0.0], dtype=np.float64),
            src_pool_soft=np.array([1.0], dtype=np.float64),
            src_pool_var=np.array([0.0], dtype=np.float64),
            src_random=np.array([0.0], dtype=np.float64),
        )

    shortlist_x = []
    shortlist_meta = []
    shortlist_by_key = {}
    cursor = {name: 0 for name in source_lists.keys()}

    def _flag_name_for_source(source_name: str) -> str:
        return {
            'softopt': 'softopt',
            'pool_soft': 'pool_soft',
            'pool_var': 'pool_var',
            'pool_random': 'random',
        }[source_name]

    def _take_from_source(source_name: str, quota: int = None):
        quota_eff = None if quota is None else int(max(0, quota))
        if (quota_eff is not None) and (quota_eff <= 0):
            return
        added = 0
        items = source_lists[source_name]
        while cursor[source_name] < len(items):
            x = np.asarray(items[cursor[source_name]], dtype=np.float32)
            cursor[source_name] += 1
            key = _key_from_x(x)
            idx_seen = shortlist_by_key.get(key)
            if idx_seen is None:
                shortlist_by_key[key] = len(shortlist_x)
                shortlist_x.append(x)
                shortlist_meta.append(dict(
                    softopt=False,
                    pool_soft=False,
                    pool_var=False,
                    random=False,
                ))
                idx_seen = len(shortlist_x) - 1
                added += 1
            shortlist_meta[idx_seen][_flag_name_for_source(source_name)] = True
            if (quota_eff is not None) and (added >= quota_eff):
                break

    _take_from_source('softopt', quota_softopt)
    _take_from_source('pool_soft', quota_pool_soft)
    _take_from_source('pool_var', quota_pool_var)
    _take_from_source('pool_random', quota_pool_random)

    if len(shortlist_x) < target_total:
        for source_name in ('softopt', 'pool_soft', 'pool_var', 'pool_random'):
            if len(shortlist_x) >= target_total:
                break
            _take_from_source(source_name, target_total - len(shortlist_x))

    if len(shortlist_x) <= 0:
        best_idx = int(np.nanargmax(U_pool)) if np.isfinite(U_pool).any() else int(rng_np.randint(0, int(X_cand.shape[0])))
        shortlist_x = [np.asarray(X_cand[best_idx], dtype=np.float32)]
        shortlist_meta = [dict(softopt=False, pool_soft=True, pool_var=False, random=False)]

    return dict(
        X=np.stack(shortlist_x, axis=0).astype(np.float32),
        src_softopt=np.asarray([float(m['softopt']) for m in shortlist_meta], dtype=np.float64),
        src_pool_soft=np.asarray([float(m['pool_soft']) for m in shortlist_meta], dtype=np.float64),
        src_pool_var=np.asarray([float(m['pool_var']) for m in shortlist_meta], dtype=np.float64),
        src_random=np.asarray([float(m['random']) for m in shortlist_meta], dtype=np.float64),
    )


# =====================================================
# Noise Config & helpers
# =====================================================
class NoiseConfig:
    def __init__(self, base_seed:int, overrides:dict=None):
        params = NOISE_DEFAULTS.copy()
        if overrides:
            params.update(overrides)
        self.control_jitter_std = float(params['control_jitter_std'])
        self.oracle_phi_noise_std = float(params['oracle_phi_noise_std'])
        self.reward_noise_std = float(params['reward_noise_std'])
        self.rng_ctrl    = np.random.RandomState(base_seed + 202)
        self.rng_phi     = np.random.RandomState(base_seed + 303)
        self.rng_reward  = np.random.RandomState(base_seed + 404)
        self.last_jittered_x = None

    def jitter_x(self, x:np.ndarray)->np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.control_jitter_std <= 0:
            self.last_jittered_x = np.asarray(x, dtype=np.float32).copy()
            return x
        xj = x + self.rng_ctrl.normal(0.0, self.control_jitter_std, size=x.shape)
        xj = np.clip(xj, 0.0, 1.0).astype(np.float32, copy=False)
        self.last_jittered_x = np.asarray(xj, dtype=np.float32).copy()
        return xj

    def add_phi_noise(self, phi:np.ndarray)->np.ndarray:
        if self.oracle_phi_noise_std <= 0:
            return phi
        noisy = phi + self.rng_phi.normal(0.0, self.oracle_phi_noise_std, size=phi.shape)
        return clip01(noisy)

    def add_reward_noise(self, r:float)->float:
        if self.reward_noise_std <= 0:
            return float(r)
        return float(r + self.rng_reward.normal(0.0, self.reward_noise_std))

@dataclass
class NonstationaryConfig:
    enabled: bool = False
    q95_queries: float = 200.0
    target_alpha: float = 0.95
    eval_pair_bank: int = 2000
    eval_forward_bank: int = 256
    forward_alpha_weight_bandwidth: float = 0.15
    forward_alpha_weight_floor: float = 1e-4

    def time_constant_queries(self) -> float:
        q95 = float(self.q95_queries)
        target = float(self.target_alpha)
        if (not np.isfinite(q95)) or q95 <= 0.0:
            raise ValueError("q95_queries must be a positive finite value.")
        if (not np.isfinite(target)) or target <= 0.0 or target >= 1.0:
            raise ValueError("target_alpha must be in (0, 1).")
        return float(-q95 / math.log(1.0 - target))

    def alpha_at_query(self, q: float) -> float:
        tau = self.time_constant_queries()
        qq = max(0.0, float(q))
        return float(np.clip(1.0 - math.exp(-qq / tau), 0.0, 1.0))


class DynamicOutcomeOracle(nn.Module):
    def __init__(self, base_model: SimulatedOutcomeModel, cfg: NonstationaryConfig, seed: int = 0):
        super().__init__()
        self.model_a = base_model
        self.cfg = cfg
        self.control_dim = int(base_model.control_dim)
        self.outcome_dim_total = int(base_model.outcome_dim_total)
        self.seed = int(seed)

        model_b_seed = int(seed) + 19001
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            np.random.seed(model_b_seed % (2**32 - 1))
            torch.manual_seed(model_b_seed)
            self.model_b = SimulatedOutcomeModel(
                self.control_dim,
                self.outcome_dim_total,
                seed=model_b_seed,
            )
        finally:
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
        self.model_b.eval()
        self.reset_state()

    def reset_state(self, seen_x=None):
        self.query_count = 0
        self.last_observed_x = None
        if seen_x is None:
            self.seen_x = []
        else:
            self.seen_x = [np.asarray(xx, dtype=np.float64).reshape(-1) for xx in seen_x]
        return self

    def transition_tau_queries(self) -> float:
        return self.cfg.time_constant_queries()

    def transition_alpha(self, query_count_override: float = None, alpha_override: float = None) -> float:
        if alpha_override is not None:
            return float(np.clip(float(alpha_override), 0.0, 1.0))
        q = self.query_count if query_count_override is None else float(query_count_override)
        return self.cfg.alpha_at_query(q)

    def current_alpha(self) -> float:
        return self.transition_alpha()

    def mean_phi(self, x: np.ndarray, alpha_override: float = None) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)
            phi_a = self.model_a(x_t).cpu().numpy().astype(np.float64)
            phi_b = self.model_b(x_t).cpu().numpy().astype(np.float64)
        alpha = self.transition_alpha(alpha_override=alpha_override)
        mean = (1.0 - alpha) * phi_a + alpha * phi_b
        return clip01(mean).astype(np.float32)

    def mean_phi_batch(self, X: np.ndarray, alpha_override: float = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            phi_a = self.model_a(X_t).cpu().numpy().astype(np.float64)
            phi_b = self.model_b(X_t).cpu().numpy().astype(np.float64)
        alpha = self.transition_alpha(alpha_override=alpha_override)
        mean = (1.0 - alpha) * phi_a + alpha * phi_b
        return clip01(mean).astype(np.float32)

    def forward_torch(self, X: torch.Tensor, alpha_override: float = None) -> torch.Tensor:
        if X.dim() == 1:
            X = X.unsqueeze(0)
        phi_a = self.model_a.forward_torch(X)
        phi_b = self.model_b.forward_torch(X)
        alpha = self.transition_alpha(alpha_override=alpha_override)
        mean = (1.0 - float(alpha)) * phi_a + float(alpha) * phi_b
        return torch.clamp(mean, 0.0, 1.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward_torch(torch.as_tensor(X, dtype=torch.float32))

    def observe_phi(self, x: np.ndarray, noise: NoiseConfig = None) -> np.ndarray:
        self.last_observed_x = np.asarray(x, dtype=np.float64).reshape(-1)
        mean = self.mean_phi(x).astype(np.float64)
        y = mean
        if noise is not None and getattr(noise, 'oracle_phi_noise_std', 0.0) > 0.0:
            y = y + noise.rng_phi.normal(0.0, float(noise.oracle_phi_noise_std), size=y.shape)
        return clip01(y).astype(np.float32)

    def advance_after_query(self, x: np.ndarray):
        self.seen_x.append(np.asarray(x, dtype=np.float64).reshape(-1))
        self.query_count += 1

    def current_session_pos(self) -> int:
        return int(self.query_count) + 1


class DynamicRewardOracle:
    def __init__(self, base_reward_oracle, drift_oracle: DynamicOutcomeOracle, true_idx, seed: int = 0):
        self.base_reward_oracle = base_reward_oracle
        self.drift_oracle = drift_oracle
        self.true_idx = np.asarray(true_idx, dtype=np.int64)

    def clean_reward(self, phi_true_slice: np.ndarray) -> float:
        with torch.no_grad():
            rr = self.base_reward_oracle(torch.tensor(np.asarray(phi_true_slice, dtype=np.float32), dtype=torch.float32))
        return float(rr.detach().cpu().numpy().reshape(-1)[0])

    def clean_reward_batch(self, Phi_true_slice: np.ndarray) -> np.ndarray:
        Phi_true_slice = np.asarray(Phi_true_slice, dtype=np.float32)
        with torch.no_grad():
            rr = self.base_reward_oracle(torch.tensor(Phi_true_slice, dtype=torch.float32))
        return rr.detach().cpu().numpy().reshape(-1).astype(np.float64)

    def reward_noise_std(self, noise: NoiseConfig = None) -> float:
        return 0.0 if noise is None else float(getattr(noise, 'reward_noise_std', 0.0))

    def observe_reward(self, phi_true_slice: np.ndarray, noise: NoiseConfig = None) -> float:
        mu = self.clean_reward(phi_true_slice)
        sigma = self.reward_noise_std(noise=noise)
        if sigma <= 0.0:
            return float(mu)
        return float(mu + noise.rng_reward.normal(0.0, sigma))

    def __call__(self, phi_tensor: torch.Tensor) -> torch.Tensor:
        phi_tensor = torch.as_tensor(phi_tensor, dtype=torch.float32)
        one = (phi_tensor.dim() == 1)
        if one:
            phi_tensor = phi_tensor.unsqueeze(0)
        out_np = self.clean_reward_batch(phi_tensor.detach().cpu().numpy())
        out = torch.tensor(out_np, dtype=phi_tensor.dtype, device=phi_tensor.device)
        return out.view(1) if one else out


def maybe_wrap_with_drift(base_outcome_oracle,
                          base_reward_oracle,
                          drift_cfg: NonstationaryConfig,
                          true_idx,
                          seed: int):
    if drift_cfg is None or (not bool(drift_cfg.enabled)):
        return base_outcome_oracle, base_reward_oracle
    drift_oracle = DynamicOutcomeOracle(base_outcome_oracle, drift_cfg, seed=int(seed) + 123)
    drift_reward = DynamicRewardOracle(base_reward_oracle, drift_oracle, true_idx=true_idx, seed=int(seed) + 321)
    return drift_oracle, drift_reward


def get_drift_oracle(outcome_model):
    return outcome_model if isinstance(outcome_model, DynamicOutcomeOracle) else None


def _executed_x_from_noise(noise: NoiseConfig, x_fallback) -> np.ndarray:
    x_exec = getattr(noise, 'last_jittered_x', None)
    if x_exec is None:
        return np.asarray(x_fallback, dtype=np.float32).copy()
    return np.asarray(x_exec, dtype=np.float32).copy()


def oracle_phi_from_x(x, outcome_model, noise:NoiseConfig):
    xj = noise.jitter_x(np.asarray(x, dtype=np.float32))
    if isinstance(outcome_model, DynamicOutcomeOracle):
        return outcome_model.observe_phi(xj, noise=noise)
    with torch.no_grad():
        phi = outcome_model(torch.tensor(xj, dtype=torch.float32)).numpy()
    return noise.add_phi_noise(phi)

def oracle_reward_from_phi(phi_true_slice, reward_oracle, noise:NoiseConfig):
    if isinstance(reward_oracle, DynamicRewardOracle):
        return reward_oracle.observe_reward(phi_true_slice, noise=noise)
    with torch.no_grad():
        r = reward_oracle(torch.tensor(phi_true_slice, dtype=torch.float32))
        r = float(r.detach().cpu().numpy().reshape(-1)[0])
    return noise.add_reward_noise(r)

def oracle_reward_from_phi_preserve_noise_state(phi_true_slice, reward_oracle, noise: NoiseConfig):
    if noise is None or not hasattr(noise, "rng_reward"):
        return oracle_reward_from_phi(phi_true_slice, reward_oracle, noise)
    reward_rng_state = noise.rng_reward.get_state()
    try:
        return oracle_reward_from_phi(phi_true_slice, reward_oracle, noise)
    finally:
        noise.rng_reward.set_state(reward_rng_state)

# =====================================================
# Initial query
# =====================================================
def generate_initial_query(reward_oracle, control_dim, outcome_dim_true, outcome_dim_est,
                           noise_cfg:NoiseConfig, net, est_indices, true_indices,
                           initial_x_pair=None):
    if initial_x_pair is None:
        x_a=np.random.rand(control_dim); x_b=np.random.rand(control_dim)
    else:
        pair = np.asarray(initial_x_pair, dtype=np.float32)
        if pair.shape != (2, int(control_dim)):
            raise ValueError(f"initial_x_pair must have shape (2, {int(control_dim)}), got {pair.shape}.")
        x_a = pair[0].copy()
        x_b = pair[1].copy()
    phi_a_full = oracle_phi_from_x(x_a, net, noise_cfg)
    x_a_exec = _executed_x_from_noise(noise_cfg, x_a)
    phi_b_full = oracle_phi_from_x(x_b, net, noise_cfg)
    x_b_exec = _executed_x_from_noise(noise_cfg, x_b)
    r_a = oracle_reward_from_phi(phi_a_full[true_indices], reward_oracle, noise_cfg)
    r_b = oracle_reward_from_phi(phi_b_full[true_indices], reward_oracle, noise_cfg)
    true_label = 1 if r_a>r_b else -1
    I = true_label
    query = (I, phi_a_full, phi_b_full)
    phi_a_est=phi_a_full[est_indices]; phi_b_est=phi_b_full[est_indices]
    return query, (x_a_exec,phi_a_est), (x_b_exec,phi_b_est)

# =====================================================
# Forward RMSE
# =====================================================
def forward_rmse_random(forward_net, outcome_model, n_test, control_dim, est_idx, noise_cfg:NoiseConfig):
    with torch.no_grad():
        X=np.random.rand(n_test,control_dim).astype(np.float32)
        pred=forward_net(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    if (noise_cfg.control_jitter_std > 0) or (noise_cfg.oracle_phi_noise_std > 0):
        true_list=[]
        for i in range(n_test):
            phi_true = oracle_phi_from_x(X[i], outcome_model, noise_cfg)
            true_list.append(phi_true[est_idx])
        true = np.stack(true_list, axis=0)
    else:
        with torch.no_grad():
            phi_full = outcome_model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
        true = phi_full[:, est_idx]
    return float(np.sqrt(((pred-true)**2).mean()))


def _noise_overrides_from_cfg(noise_cfg: NoiseConfig) -> dict:
    return dict(
        control_jitter_std=float(noise_cfg.control_jitter_std),
        oracle_phi_noise_std=float(noise_cfg.oracle_phi_noise_std),
        reward_noise_std=float(noise_cfg.reward_noise_std),
    )


def _forward_rmse_seeded(forward_net,
                         outcome_model,
                         n_test: int,
                         control_dim: int,
                         est_idx,
                         noise_cfg: NoiseConfig,
                         *,
                         seed: int) -> float:
    n_test = int(max(1, n_test))
    rng = np.random.RandomState(int(seed))
    X = rng.rand(n_test, int(control_dim)).astype(np.float32)
    with torch.no_grad():
        pred = forward_net(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    # RMSE is defined against the clean oracle mean, not noisy observations.
    if isinstance(outcome_model, DynamicOutcomeOracle):
        true = outcome_model.mean_phi_batch(X).astype(np.float64)[:, est_idx]
    else:
        with torch.no_grad():
            phi_full = outcome_model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
        true = np.asarray(phi_full[:, est_idx], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def build_control_pair_bank(control_dim: int, n_pairs: int, seed: int):
    rs = np.random.RandomState(int(seed))
    XA = rs.rand(int(n_pairs), int(control_dim)).astype(np.float32)
    XB = rs.rand(int(n_pairs), int(control_dim)).astype(np.float32)
    return XA, XB


def _take_evenly(idx: np.ndarray, take: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    take = int(max(0, take))
    if take <= 0 or idx.size <= 0:
        return np.zeros((0,), dtype=np.int64)
    if idx.size <= take:
        return idx
    sel = np.linspace(0, idx.size - 1, num=take, dtype=int)
    return idx[sel]


def _balanced_counts_for_total(available_counts, total_target: int) -> np.ndarray:
    available = np.maximum(np.asarray(available_counts, dtype=np.int64), 0)
    n_bins = int(available.size)
    total_target = int(max(0, total_target))
    if n_bins <= 0 or total_target <= 0:
        return np.zeros((n_bins,), dtype=np.int64)

    total = min(total_target, int(np.sum(available)))
    base = total // n_bins
    rem = total % n_bins
    counts = np.array([base + (1 if i < rem else 0) for i in range(n_bins)], dtype=np.int64)
    counts = np.minimum(counts, available)

    deficit = int(total - np.sum(counts))
    while deficit > 0:
        capacity = available - counts
        spare_idx = np.flatnonzero(capacity > 0)
        if spare_idx.size <= 0:
            break
        add_base = deficit // int(spare_idx.size)
        add_rem = deficit % int(spare_idx.size)
        added = 0
        for pos, j in enumerate(spare_idx):
            want = add_base + (1 if pos < add_rem else 0)
            if want <= 0:
                continue
            add = min(int(want), int(capacity[j]))
            counts[j] += add
            added += add
        if added <= 0:
            break
        deficit -= added

    return counts


def _forward_rmse_drift_bank(forward_net,
                             drift_oracle: DynamicOutcomeOracle,
                             X_eval: np.ndarray,
                             est_idx) -> float:
    X_eval = np.asarray(X_eval, dtype=np.float32)
    with torch.no_grad():
        pred = forward_net(torch.tensor(X_eval, dtype=torch.float32)).cpu().numpy()
    true = drift_oracle.mean_phi_batch(X_eval).astype(np.float64)[:, est_idx]
    return float(np.sqrt(np.mean((np.asarray(pred, dtype=np.float64) - true) ** 2)))


def _build_current_drift_eval_sets(drift_oracle: DynamicOutcomeOracle,
                                   reward_oracle: DynamicRewardOracle,
                                   XA_bank: np.ndarray,
                                   XB_bank: np.ndarray,
                                   est_idx,
                                   true_idx,
                                   tau_list,
                                   *,
                                   target_kept: int,
                                   balanced_per_bin: int = 0,
                                   balanced_total: int = None,
                                   min_outcome_axis_gap: float = None):
    PhiA_full = drift_oracle.mean_phi_batch(XA_bank).astype(np.float64)
    PhiB_full = drift_oracle.mean_phi_batch(XB_bank).astype(np.float64)
    PhiA = PhiA_full[:, est_idx].astype(np.float64, copy=False)
    PhiB = PhiB_full[:, est_idx].astype(np.float64, copy=False)

    rA = reward_oracle.clean_reward_batch(PhiA_full[:, true_idx])
    rB = reward_oracle.clean_reward_batch(PhiB_full[:, true_idx])
    gap = np.asarray(rA - rB, dtype=np.float64)
    abs_gap = np.abs(gap)
    y_clean = np.where(gap >= 0.0, 1, -1).astype(np.int64)

    mask_base = np.ones((abs_gap.shape[0],), dtype=bool)
    if min_outcome_axis_gap is not None and float(min_outcome_axis_gap) > 0.0:
        max_axis_gap = np.max(np.abs(PhiA - PhiB), axis=1)
        mask_base &= (max_axis_gap >= float(min_outcome_axis_gap))

    tau_sorted = [float(t) for t in sorted(list(tau_list))]
    edges = tau_sorted + [float('inf')]
    by_tau = {}
    idx_by_bin = []
    for kk, tau in enumerate(tau_sorted):
        hi = float(edges[kk + 1])
        idx = np.flatnonzero(mask_base & (abs_gap >= float(tau)) & (abs_gap < hi))
        idx_by_bin.append(idx)
        idx_tau = _take_evenly(idx, min(int(target_kept), int(idx.size)))
        by_tau[tau] = (
            np.asarray(PhiA[idx_tau], dtype=np.float32),
            np.asarray(PhiB[idx_tau], dtype=np.float32),
            np.asarray(y_clean[idx_tau], dtype=np.int64),
        )

    if balanced_total is not None:
        balanced_counts = _balanced_counts_for_total(
            [idx.size for idx in idx_by_bin],
            int(balanced_total),
        )
    else:
        per = int(max(0, balanced_per_bin))
        balanced_counts = np.asarray([min(per, int(idx.size)) for idx in idx_by_bin], dtype=np.int64)

    balanced_idx_parts = []
    for idx, take in zip(idx_by_bin, balanced_counts):
        idx_bal = _take_evenly(idx, int(take))
        if idx_bal.size > 0:
            balanced_idx_parts.append(idx_bal)

    if balanced_idx_parts:
        idx_balanced = np.concatenate(balanced_idx_parts, axis=0)
        balanced_pack = (
            np.asarray(PhiA[idx_balanced], dtype=np.float32),
            np.asarray(PhiB[idx_balanced], dtype=np.float32),
            np.asarray(y_clean[idx_balanced], dtype=np.int64),
        )
    else:
        balanced_pack = None
    return by_tau, balanced_pack


# =====================================================
# Self-contained RBF-RFF latent utility GP over outcomes
# =====================================================
@dataclass
class PairwiseRFFGPConfig:
    d_est: int
    m_features: int = INVERSE_GP_RFF_M
    lengthscale: float = INVERSE_GP_RBF_ELL
    prior_std_w: float = INVERSE_GP_PRIOR_STD
    eta_mu: float = INVERSE_GP_ETA_MU
    eta_sig: float = INVERSE_GP_ETA_SIG
    cov_jitter: float = INVERSE_GP_COV_JITTER
    map_lbfgs_iters: int = INVERSE_GP_MAP_LBFGS_ITERS
    posterior_samples: int = INVERSE_GP_POST_SAMPLES
    seed: int = 0
    feature_type: str = "rff"
    phi_mean: object = None
    phi_std: object = None

    def build_features(self):
        feature_type = str(self.feature_type).lower().strip()
        if feature_type in ("linear", "linear_laplace"):
            return LinearStandardizedFeatures(
                d_est=int(self.d_est),
                phi_mean=self.phi_mean,
                phi_std=self.phi_std,
            )
        if feature_type not in ("rff", "gp_rff"):
            raise ValueError(f"Unknown inverse feature_type={self.feature_type!r}.")
        rs = np.random.RandomState(int(self.seed) + 7001)
        omega = rs.normal(0.0, 1.0, size=(self.m_features, self.d_est)).astype(np.float32)
        phase = rs.uniform(0.0, 2.0 * np.pi, size=(self.m_features,)).astype(np.float32)
        return RFFFeatures(omega=omega, phase=phase, lengthscale=float(self.lengthscale))


class RFFFeatures:
    def __init__(self, omega: np.ndarray, phase: np.ndarray, lengthscale: float):
        self.omega = np.asarray(omega, dtype=np.float32)
        self.phase = np.asarray(phase, dtype=np.float32)
        self.lengthscale = float(lengthscale)
        self.output_dim = int(self.omega.shape[0])
        self.sqrt2_over_m = math.sqrt(2.0 / float(self.omega.shape[0]))

    def transform_np(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=np.float32)
        if phi.ndim == 1:
            phi = phi[None, :]
        proj = (phi / self.lengthscale) @ self.omega.T
        z = np.cos(proj + self.phase[None, :]) * self.sqrt2_over_m
        return z.astype(np.float64)

    def transform_torch(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        omega_t = torch.as_tensor(self.omega, dtype=phi.dtype, device=phi.device)
        phase_t = torch.as_tensor(self.phase, dtype=phi.dtype, device=phi.device)
        proj = (phi / float(self.lengthscale)) @ omega_t.t()
        z = torch.cos(proj + phase_t.unsqueeze(0)) * float(self.sqrt2_over_m)
        return z


class LinearStandardizedFeatures:
    def __init__(self, d_est: int, phi_mean=None, phi_std=None):
        self.d_est = int(d_est)
        self.output_dim = int(d_est)
        if phi_mean is None:
            phi_mean = np.full((self.d_est,), 0.5, dtype=np.float64)
        if phi_std is None:
            phi_std = np.full((self.d_est,), np.sqrt(1.0 / 12.0), dtype=np.float64)
        self.phi_mean = np.asarray(phi_mean, dtype=np.float64).reshape(-1)
        self.phi_std = np.maximum(np.asarray(phi_std, dtype=np.float64).reshape(-1), 1e-6)
        if self.phi_mean.size != self.d_est or self.phi_std.size != self.d_est:
            raise ValueError("LinearStandardizedFeatures phi_mean/std must match d_est.")

    def transform_np(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=np.float32)
        if phi.ndim == 1:
            phi = phi[None, :]
        z = (phi.astype(np.float64) - self.phi_mean[None, :]) / self.phi_std[None, :]
        return z.astype(np.float64, copy=False)

    def transform_torch(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.dim() == 1:
            phi = phi.unsqueeze(0)
        mean_t = torch.as_tensor(self.phi_mean, dtype=phi.dtype, device=phi.device)
        std_t = torch.as_tensor(self.phi_std, dtype=phi.dtype, device=phi.device)
        return (phi - mean_t.unsqueeze(0)) / std_t.unsqueeze(0)


class PairwiseRFFGPState:
    def __init__(self, cfg: PairwiseRFFGPConfig):
        self.cfg = cfg
        self.feat = cfg.build_features()
        feature_dim = int(getattr(self.feat, "output_dim", cfg.m_features))
        self.map_q = np.zeros((feature_dim + 1,), dtype=np.float64)
        self.cov = np.eye(feature_dim + 1, dtype=np.float64)
        self.posterior_samples = np.zeros((0, feature_dim + 1), dtype=np.float64)
        self.posterior_beta = np.zeros((0,), dtype=np.float64)
        self.diag = {}


def _softplus_neg_margin(margin: torch.Tensor) -> torch.Tensor:
    return F.softplus(-margin)


def _symmetrize_psd(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    return 0.5 * (mat + mat.T)


def _safe_gaussian_samples(mean: np.ndarray,
                           cov: np.ndarray,
                           size: int,
                           rs: np.random.RandomState,
                           *,
                           eig_floor: float = 1e-10):
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    cov = _symmetrize_psd(cov)
    dim = int(mean.shape[0])

    if cov.shape != (dim, dim):
        raise ValueError(f'cov shape {cov.shape} does not match mean dim {dim}')

    if (not np.all(np.isfinite(mean))) or (not np.all(np.isfinite(cov))):
        cov = np.eye(dim, dtype=np.float64)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        cov = _symmetrize_psd(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
        cov += float(eig_floor) * np.eye(dim, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(cov)

    eigvals = np.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals_clipped = np.clip(eigvals, float(eig_floor), None)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals_clipped))
    z = rs.randn(int(size), dim).astype(np.float64)
    samples = mean[None, :] + z @ sqrt_cov.T
    diag = dict(
        min_eig=float(np.min(eigvals)) if eigvals.size > 0 else float('nan'),
        min_eig_clipped=float(np.min(eigvals_clipped)) if eigvals_clipped.size > 0 else float('nan'),
        clipped_eigs=int(np.sum(eigvals < eig_floor)),
    )
    return samples.astype(np.float64, copy=False), diag


def _stable_cov_from_precision(H: np.ndarray,
                               *,
                               base_floor: float = 1e-10):
    H = _symmetrize_psd(H)
    dim = int(H.shape[0])
    if H.shape[1] != dim:
        raise ValueError(f'precision matrix must be square, got {H.shape}')

    if not np.all(np.isfinite(H)):
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H += np.eye(dim, dtype=np.float64)

    try:
        eigvals, eigvecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        H = _symmetrize_psd(H + float(base_floor) * np.eye(dim, dtype=np.float64))
        eigvals, eigvecs = np.linalg.eigh(H)

    eigvals = np.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(np.abs(eigvals))) if eigvals.size > 0 else 1.0
    floor = max(float(base_floor), 1e-12 * max(scale, 1.0))
    eigvals_clipped = np.clip(eigvals, floor, None)
    inv_eigs = 1.0 / eigvals_clipped
    cov = eigvecs @ np.diag(inv_eigs) @ eigvecs.T
    cov = _symmetrize_psd(cov)
    diag = dict(
        min_eig_H=float(np.min(eigvals)) if eigvals.size > 0 else float('nan'),
        min_eig_H_clipped=float(np.min(eigvals_clipped)) if eigvals_clipped.size > 0 else float('nan'),
        clipped_eigs_H=int(np.sum(eigvals < floor)),
        cov_trace=float(np.trace(cov)),
    )
    return cov.astype(np.float64, copy=False), diag


def fit_pairwise_rff_gp(phiA: np.ndarray,
                        phiB: np.ndarray,
                        y_pm1: np.ndarray,
                        gp_state: PairwiseRFFGPState,
                        *,
                        warm_start: bool = True,
                        seed: int = 0):
    cfg = gp_state.cfg
    phiA = np.asarray(phiA, dtype=np.float32)
    phiB = np.asarray(phiB, dtype=np.float32)
    y = np.asarray(y_pm1, dtype=np.float32).reshape(-1)

    if phiA.ndim == 1:
        phiA = phiA[None, :]
    if phiB.ndim == 1:
        phiB = phiB[None, :]

    ZA = gp_state.feat.transform_np(phiA)
    ZB = gp_state.feat.transform_np(phiB)
    X = ZA - ZB
    N, M = X.shape

    if N == 0:
        gp_state.map_q = np.zeros((M + 1,), dtype=np.float64)
        gp_state.cov = np.eye(M + 1, dtype=np.float64)
        rs = np.random.RandomState(int(seed) + 91)
        gp_state.posterior_samples, samp_diag = _safe_gaussian_samples(
            mean=gp_state.map_q,
            cov=gp_state.cov,
            size=int(cfg.posterior_samples),
            rs=rs,
            eig_floor=max(float(cfg.cov_jitter), 1e-10),
        )
        gp_state.posterior_beta = np.exp(gp_state.posterior_samples[:, -1])
        gp_state.diag = dict(
            n_obs=0,
            beta_mean=float(np.mean(gp_state.posterior_beta)),
            **samp_diag,
        )
        return gp_state

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    if warm_start and gp_state.map_q.shape[0] == (M + 1):
        q0 = gp_state.map_q.copy()
    else:
        q0 = np.zeros((M + 1,), dtype=np.float64)

    q = torch.tensor(q0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([q], lr=0.5, max_iter=int(cfg.map_lbfgs_iters), line_search_fn='strong_wolfe')

    prior_prec_w = 1.0 / max(float(cfg.prior_std_w) ** 2, 1e-12)
    prior_prec_eta = 1.0 / max(float(cfg.eta_sig) ** 2, 1e-12)

    def closure():
        opt.zero_grad()
        w = q[:-1]
        eta = q[-1]
        beta = torch.exp(torch.clamp(eta, min=-20.0, max=20.0))
        delta = X_t @ w
        margin = y_t * beta * delta
        neg_log_likelihood = _softplus_neg_margin(margin).sum()
        prior = 0.5 * prior_prec_w * torch.sum(w * w) + 0.5 * prior_prec_eta * ((eta - float(cfg.eta_mu)) ** 2)
        obj = neg_log_likelihood + prior
        obj.backward()
        return obj

    opt.step(closure)

    with torch.no_grad():
        q_map = q.detach().cpu().numpy().astype(np.float64)
        w_map = q_map[:-1]
        eta_map = float(q_map[-1])
        beta_map = float(np.exp(np.clip(eta_map, -20.0, 20.0)))

        delta_map = X @ w_map
        a = y.astype(np.float64) * beta_map * delta_map
        s = 1.0 / (1.0 + np.exp(-np.clip(a, -60.0, 60.0)))
        lam = np.clip(s * (1.0 - s), 1e-8, None)

        gw = (y[:, None].astype(np.float64) * beta_map) * X
        geta = y.astype(np.float64) * beta_map * delta_map
        G = np.concatenate([gw, geta[:, None]], axis=1)

        H = np.diag(np.concatenate([
            np.full((M,), prior_prec_w, dtype=np.float64),
            np.array([prior_prec_eta], dtype=np.float64),
        ]))
        H += G.T @ (lam[:, None] * G)
        H += float(cfg.cov_jitter) * np.eye(M + 1, dtype=np.float64)

        cov, cov_diag = _stable_cov_from_precision(H, base_floor=max(float(cfg.cov_jitter), 1e-10))
        rs = np.random.RandomState(int(seed) + 9203 + N)
        post_samps, samp_diag = _safe_gaussian_samples(
            mean=q_map,
            cov=cov,
            size=int(cfg.posterior_samples),
            rs=rs,
            eig_floor=max(float(cfg.cov_jitter), 1e-10),
        )
        beta_samps = np.exp(np.clip(post_samps[:, -1], -10.0, 10.0))

    gp_state.map_q = q_map
    gp_state.cov = cov
    gp_state.posterior_samples = post_samps
    gp_state.posterior_beta = beta_samps
    gp_state.diag = dict(
        n_obs=int(N),
        beta_mean=float(np.mean(beta_samps)),
        beta_map=float(beta_map),
        eta_map=float(eta_map),
        **cov_diag,
        **samp_diag,
    )
    return gp_state


def gp_reward_vectorized(q_samples: np.ndarray, phi_batch: np.ndarray, gp_state: PairwiseRFFGPState) -> np.ndarray:
    phi = np.asarray(phi_batch, dtype=np.float32)
    if phi.ndim == 1:
        phi = phi[None, :]
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim == 1:
        q_samples = q_samples[None, :]
    Z = gp_state.feat.transform_np(phi)
    W = q_samples[:, :-1]
    return W @ Z.T


def gp_reward_vectorized_torch(q_samples: torch.Tensor,
                               phi_batch: torch.Tensor,
                               gp_state: PairwiseRFFGPState) -> torch.Tensor:
    if phi_batch.dim() == 1:
        phi_batch = phi_batch.unsqueeze(0)
    if q_samples.dim() == 1:
        q_samples = q_samples.unsqueeze(0)
    Z = gp_state.feat.transform_torch(phi_batch)
    W = q_samples[:, :-1].to(device=phi_batch.device, dtype=phi_batch.dtype)
    return W @ Z.t()


def compute_predictive_pbar_from_phi_array_gp(q_samples: np.ndarray,
                                              phi_anchor: np.ndarray,
                                              phi_targets: np.ndarray,
                                              gp_state: PairwiseRFFGPState,
                                              *,
                                              noise_cfg=None,
                                              reward_weights=None):
    phi_targets = np.asarray(phi_targets, dtype=np.float32)
    if phi_targets.ndim == 1:
        phi_targets = phi_targets[None, :]
    phi_anchor = np.asarray(phi_anchor, dtype=np.float32).reshape(1, -1)
    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim == 1:
        q_samples = q_samples[None, :]

    beta = np.exp(np.clip(q_samples[:, -1], -10.0, 10.0)).reshape(-1, 1)
    r_anchor = gp_reward_vectorized(q_samples, phi_anchor, gp_state).reshape(-1, 1)
    r_tgt = gp_reward_vectorized(q_samples, phi_targets, gp_state)
    logits = beta * (r_anchor - r_tgt)
    p_s = _pref_probs_from_logits(logits, beta=beta, noise_cfg=noise_cfg)

    if reward_weights is None:
        pbar = np.mean(p_s, axis=0)
    else:
        w = np.asarray(reward_weights, dtype=np.float64).reshape(-1, 1)
        wsum = float(np.sum(w))
        if wsum <= 1e-12 or (not np.isfinite(wsum)):
            pbar = np.mean(p_s, axis=0)
        else:
            pbar = np.sum(w * p_s, axis=0) / wsum
    return np.asarray(pbar, dtype=np.float64)


def conditional_soft_uncertainty_from_phi_samples_gp(q_samples: np.ndarray,
                                                     phi_anchor: np.ndarray,
                                                     Phi_samp: np.ndarray,
                                                     gp_state: PairwiseRFFGPState,
                                                     *,
                                                     noise_cfg=None,
                                                     reward_weights=None):
    Phi_samp = np.asarray(Phi_samp, dtype=np.float32)
    if Phi_samp.ndim == 2:
        Phi_samp = Phi_samp[None, :, :]
    acc = 0.0
    for m in range(Phi_samp.shape[0]):
        q = compute_predictive_pbar_from_phi_array_gp(
            q_samples, phi_anchor, Phi_samp[m], gp_state,
            noise_cfg=noise_cfg,
            reward_weights=reward_weights,
        )
        acc += float(np.mean(4.0 * q * (1.0 - q)))
    return float(acc / float(max(1, Phi_samp.shape[0])))


def conditional_mi_from_phi_samples_gp(q_samples: np.ndarray,
                                       phi_anchor: np.ndarray,
                                       Phi_samp: np.ndarray,
                                       gp_state: PairwiseRFFGPState,
                                       *,
                                       noise_cfg=None,
                                       reward_weights=None):
    Phi_samp = np.asarray(Phi_samp, dtype=np.float32)
    if Phi_samp.ndim == 2:
        Phi_samp = Phi_samp[None, :, :]

    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim == 1:
        q_samples = q_samples[None, :]

    beta = np.exp(np.clip(q_samples[:, -1], -10.0, 10.0)).reshape(-1, 1)
    r_anchor = gp_reward_vectorized(
        q_samples,
        np.asarray(phi_anchor, dtype=np.float32).reshape(1, -1),
        gp_state,
    ).reshape(-1, 1)

    w = None
    if reward_weights is not None:
        w_try = np.asarray(reward_weights, dtype=np.float64).reshape(-1, 1)
        if w_try.shape[0] == q_samples.shape[0]:
            wsum = float(np.sum(w_try))
            if np.isfinite(wsum) and wsum > 1e-12:
                w = w_try / wsum

    M_eff = int(max(1, Phi_samp.shape[0]))
    pbar_m = []
    H_cond_acc = None

    for m in range(Phi_samp.shape[0]):
        r_tgt = gp_reward_vectorized(q_samples, Phi_samp[m], gp_state)
        logits = beta * (r_anchor - r_tgt)
        p_s = _pref_probs_from_logits(logits, beta=beta, noise_cfg=noise_cfg)

        if w is None:
            pbar_mt = np.mean(p_s, axis=0)
            H_cond_m = np.mean(bern_entropy_np(p_s), axis=0)
        else:
            pbar_mt = np.sum(w * p_s, axis=0)
            H_cond_m = np.sum(w * bern_entropy_np(p_s), axis=0)

        if H_cond_acc is None:
            H_cond_acc = np.zeros_like(H_cond_m, dtype=np.float64)
        H_cond_acc += np.asarray(H_cond_m, dtype=np.float64)
        pbar_m.append(np.asarray(pbar_mt, dtype=np.float64))

    pbar_m = np.asarray(pbar_m, dtype=np.float64)
    pbar = np.mean(pbar_m, axis=0)
    H_bar = bern_entropy_np(pbar)
    E_H = np.asarray(H_cond_acc, dtype=np.float64) / float(M_eff)
    mi = np.maximum(H_bar - E_H, 0.0)
    return float(np.mean(mi))


def compute_anchor_soft_scores_from_mean_phi_gp(phi_cand: np.ndarray,
                                                phi_anchor: np.ndarray,
                                                q_samples: np.ndarray,
                                                gp_state: PairwiseRFFGPState,
                                                *,
                                                noise_cfg=None):
    phi_cand = np.asarray(phi_cand, dtype=np.float32)
    if phi_cand.ndim == 1:
        phi_cand = phi_cand[None, :]
    q_samples = np.asarray(q_samples, dtype=np.float64)
    beta = np.exp(np.clip(q_samples[:, -1], -10.0, 10.0)).reshape(-1, 1)
    r_anchor = gp_reward_vectorized(q_samples, np.asarray(phi_anchor, dtype=np.float32).reshape(1, -1), gp_state).reshape(-1, 1)
    r_cand = gp_reward_vectorized(q_samples, phi_cand, gp_state)
    logits = beta * (r_anchor - r_cand)
    p_s = _pref_probs_from_logits(logits, beta=beta, noise_cfg=noise_cfg)
    pbar = np.mean(p_s, axis=0)
    U = 4.0 * pbar * (1.0 - pbar)
    return dict(score=U.astype(np.float64), pbar=pbar.astype(np.float64), p_s=p_s.astype(np.float64, copy=False))


def compute_anchor_bald_scores_from_mean_phi_gp(phi_cand: np.ndarray,
                                                phi_anchor: np.ndarray,
                                                q_samples: np.ndarray,
                                                gp_state: PairwiseRFFGPState,
                                                *,
                                                noise_cfg=None):
    soft = compute_anchor_soft_scores_from_mean_phi_gp(
        phi_cand, phi_anchor, q_samples, gp_state, noise_cfg=noise_cfg,
    )
    pbar = np.asarray(soft['pbar'], dtype=np.float64)
    p_s = np.asarray(soft['p_s'], dtype=np.float64)
    H_bar = bern_entropy_np(pbar)
    E_H = np.mean(bern_entropy_np(p_s), axis=0)
    bald = H_bar - E_H
    return dict(
        score=np.asarray(bald, dtype=np.float64),
        pbar=pbar,
        p_s=p_s,
        soft_score=np.asarray(soft['score'], dtype=np.float64),
    )


def compute_cond_mi_anchor_scores_from_candidates_gp(X_cand: np.ndarray,
                                                     phi_cand_mean: np.ndarray,
                                                     phi_anchor: np.ndarray,
                                                     forward_net,
                                                     q_samples: np.ndarray,
                                                     gp_state: PairwiseRFFGPState,
                                                     *,
                                                     rng_np=None,
                                                     n_fwd_samples: int = JOINT_FWD_SAMPLES,
                                                     noise_cfg=None,
                                                     phi_cand_var_total=None):
    if rng_np is None:
        rng_np = np.random

    X_cand = np.asarray(X_cand, dtype=np.float32)
    if X_cand.ndim == 1:
        X_cand = X_cand[None, :]

    phi_cand_mean = np.asarray(phi_cand_mean, dtype=np.float64)
    if phi_cand_mean.ndim == 1:
        phi_cand_mean = phi_cand_mean[None, :]

    q_samples = np.asarray(q_samples, dtype=np.float64)
    if q_samples.ndim == 1:
        q_samples = q_samples[None, :]

    M_req = int(max(1, n_fwd_samples))
    if noise_cfg is not None:
        if phi_cand_var_total is None:
            fwd_pack = get_forward_mean_var_decomposed_np(forward_net, X_cand, noise_cfg=noise_cfg)
            phi_sample_mean = np.asarray(fwd_pack['mean'], dtype=np.float64)
            phi_sample_var = np.asarray(fwd_pack['var_total'], dtype=np.float64)
        else:
            phi_sample_mean = phi_cand_mean
            phi_sample_var = np.asarray(phi_cand_var_total, dtype=np.float64)
        Phi_samp = sample_phi_from_mean_var_np(
            phi_sample_mean,
            phi_sample_var,
            M_req,
            rng_np=rng_np,
        )
    elif hasattr(forward_net, 'sample_phi') and int(getattr(forward_net, 'n_seen', 0)) > 0:
        Phi_samp = np.asarray(
            forward_net.sample_phi(X_cand, n_samples=M_req, rng_np=rng_np),
            dtype=np.float64,
        )
    else:
        Phi_samp = np.repeat(phi_cand_mean[None, :, :], M_req, axis=0)

    if Phi_samp.ndim == 2:
        Phi_samp = Phi_samp[None, :, :]

    beta = np.exp(np.clip(q_samples[:, -1], -10.0, 10.0)).reshape(-1, 1)
    r_anchor = gp_reward_vectorized(
        q_samples,
        np.asarray(phi_anchor, dtype=np.float32).reshape(1, -1),
        gp_state,
    ).reshape(-1, 1)

    M_eff, N_eval, _ = Phi_samp.shape
    pbar_m = np.empty((M_eff, N_eval), dtype=np.float64)
    H_ps_acc = np.zeros((N_eval,), dtype=np.float64)

    for m in range(M_eff):
        r_cand_m = gp_reward_vectorized(q_samples, np.asarray(Phi_samp[m], dtype=np.float32), gp_state)
        logits_m = beta * (r_anchor - r_cand_m)
        p_s_m = _pref_probs_from_logits(logits_m, beta=beta, noise_cfg=noise_cfg)
        pbar_m[m, :] = np.mean(p_s_m, axis=0)
        H_ps_acc += np.mean(bern_entropy_np(p_s_m), axis=0)

    pbar = np.mean(pbar_m, axis=0)
    H_bar = bern_entropy_np(pbar)
    H_bar_m = np.mean(bern_entropy_np(pbar_m), axis=0)
    E_H = H_ps_acc / float(max(1, M_eff))
    score = np.maximum(H_bar - E_H, 0.0)
    aux_fwd = np.maximum(H_bar - H_bar_m, 0.0)
    aux_inv = np.maximum(H_bar_m - E_H, 0.0)
    return dict(
        score=np.asarray(score, dtype=np.float64),
        pbar=np.asarray(pbar, dtype=np.float64),
        aux_fwd=np.asarray(aux_fwd, dtype=np.float64),
        aux_inv=np.asarray(aux_inv, dtype=np.float64),
    )


def build_target_set_soft_pool_gp(X_cand: np.ndarray,
                                  phi_cand_mean: np.ndarray,
                                  phi_anchor: np.ndarray,
                                  q_samples: np.ndarray,
                                  gp_state: PairwiseRFFGPState,
                                  *,
                                  topk: int = BOUNDARY_LOOKAHEAD_TARGET_TOPK):
    stats = compute_anchor_soft_scores_from_mean_phi_gp(
        phi_cand_mean, phi_anchor, q_samples, gp_state,
    )
    score = np.asarray(stats['score'], dtype=np.float64)
    topk = int(max(1, min(int(topk), int(score.shape[0]))))
    idx = np.argsort(-score)[:topk]
    return (
        np.asarray(X_cand[idx], dtype=np.float32),
        np.asarray(phi_cand_mean[idx], dtype=np.float64),
        np.asarray(score[idx], dtype=np.float64),
        np.asarray(stats['pbar'], dtype=np.float64)[idx],
    )


def build_boundary_lookahead_shortlist_gp_soft(X_cand: np.ndarray,
                                        score_main: np.ndarray,
                                        var_score: np.ndarray,
                                        q_samples: np.ndarray,
                                        phi_anchor: np.ndarray,
                                        gp_state: PairwiseRFFGPState,
                                        *,
                                        control_dim: int,
                                        forward_net,
                                        outcome_model,
                                        est_indices,
                                        device=None,
                                        rng_np=None):
    if rng_np is None:
        rng_np = np.random

    quota_opt = 0
    quota_pool_main = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_KEEP + BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_SOFT_EXTRA))
    quota_pool_var = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_VARIANCE_EXTRA))
    quota_pool_random = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_RANDOM_EXTRA))
    target_total = int(max(1, quota_opt + quota_pool_main + quota_pool_var + quota_pool_random))

    def _key_from_x(x):
        return tuple(np.round(np.asarray(x, dtype=np.float64), 6))

    source_lists = dict(
        softopt=[],
        pool_soft=[],
        pool_var=[],
        pool_random=[],
    )

    # Pool-only shortlist: skip soft-optimization proposals and spend the budget on ranked pool picks.

    score_main = np.asarray(score_main, dtype=np.float64).reshape(-1)
    ord_main = np.argsort(-np.nan_to_num(score_main, nan=-1e18))
    source_lists['pool_soft'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in ord_main]

    var_score = np.asarray(var_score, dtype=np.float64).reshape(-1)
    ord_var = np.argsort(-np.nan_to_num(var_score, nan=-1e18))
    source_lists['pool_var'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in ord_var]

    if int(X_cand.shape[0]) > 0:
        perm_rand = np.asarray(rng_np.permutation(int(X_cand.shape[0])), dtype=np.int64).reshape(-1)
        source_lists['pool_random'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in perm_rand]

    if all(len(v) <= 0 for v in source_lists.values()):
        best_idx = int(np.nanargmax(score_main)) if np.isfinite(score_main).any() else int(rng_np.randint(0, int(X_cand.shape[0])))
        x_best = np.asarray(X_cand[best_idx:best_idx + 1], dtype=np.float32)
        return dict(
            X=x_best,
            src_softopt=np.array([0.0], dtype=np.float64),
            src_pool_soft=np.array([1.0], dtype=np.float64),
            src_pool_var=np.array([0.0], dtype=np.float64),
            src_random=np.array([0.0], dtype=np.float64),
        )

    shortlist_x = []
    shortlist_meta = []
    shortlist_by_key = {}
    cursor = {name: 0 for name in source_lists.keys()}

    def _flag_name_for_source(source_name: str) -> str:
        return {
            'softopt': 'softopt',
            'pool_soft': 'pool_soft',
            'pool_var': 'pool_var',
            'pool_random': 'random',
        }[source_name]

    def _take_from_source(source_name: str, quota: int = None):
        quota_eff = None if quota is None else int(max(0, quota))
        if (quota_eff is not None) and (quota_eff <= 0):
            return
        added = 0
        items = source_lists[source_name]
        while cursor[source_name] < len(items):
            x = np.asarray(items[cursor[source_name]], dtype=np.float32)
            cursor[source_name] += 1
            key = _key_from_x(x)
            idx_seen = shortlist_by_key.get(key)
            if idx_seen is None:
                shortlist_by_key[key] = len(shortlist_x)
                shortlist_x.append(x)
                shortlist_meta.append(dict(
                    softopt=False,
                    pool_soft=False,
                    pool_var=False,
                    random=False,
                ))
                idx_seen = len(shortlist_x) - 1
                added += 1
            shortlist_meta[idx_seen][_flag_name_for_source(source_name)] = True
            if (quota_eff is not None) and (added >= quota_eff):
                break

    _take_from_source('softopt', quota_opt)
    _take_from_source('pool_soft', quota_pool_main)
    _take_from_source('pool_var', quota_pool_var)
    _take_from_source('pool_random', quota_pool_random)

    if len(shortlist_x) < target_total:
        for source_name in ('softopt', 'pool_soft', 'pool_var', 'pool_random'):
            if len(shortlist_x) >= target_total:
                break
            _take_from_source(source_name, target_total - len(shortlist_x))

    if len(shortlist_x) <= 0:
        best_idx = int(np.nanargmax(score_main)) if np.isfinite(score_main).any() else int(rng_np.randint(0, int(X_cand.shape[0])))
        shortlist_x = [np.asarray(X_cand[best_idx], dtype=np.float32)]
        shortlist_meta = [dict(softopt=False, pool_soft=True, pool_var=False, random=False)]

    return dict(
        X=np.stack(shortlist_x, axis=0).astype(np.float32),
        src_softopt=np.asarray([float(m['softopt']) for m in shortlist_meta], dtype=np.float64),
        src_pool_soft=np.asarray([float(m['pool_soft']) for m in shortlist_meta], dtype=np.float64),
        src_pool_var=np.asarray([float(m['pool_var']) for m in shortlist_meta], dtype=np.float64),
        src_random=np.asarray([float(m['random']) for m in shortlist_meta], dtype=np.float64),
    )


def build_boundary_lookahead_shortlist_gp_mi(X_cand: np.ndarray,
                                      score_main: np.ndarray,
                                      var_score: np.ndarray,
                                      q_samples: np.ndarray,
                                      phi_anchor: np.ndarray,
                                      gp_state: PairwiseRFFGPState,
                                      *,
                                      control_dim: int,
                                      forward_net,
                                      outcome_model,
                                      est_indices,
                                      device=None,
                                      rng_np=None):
    if rng_np is None:
        rng_np = np.random

    quota_opt = 0
    quota_pool_main = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_KEEP + BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_SOFT_EXTRA))
    quota_pool_var = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_VARIANCE_EXTRA))
    quota_pool_random = int(max(0, BOUNDARY_LOOKAHEAD_SHORTLIST_POOL_RANDOM_EXTRA))
    target_total = int(max(1, quota_opt + quota_pool_main + quota_pool_var + quota_pool_random))

    def _key_from_x(x):
        return tuple(np.round(np.asarray(x, dtype=np.float64), 6))

    source_lists = dict(
        softopt=[],
        pool_soft=[],
        pool_var=[],
        pool_random=[],
    )

    # Pool-only shortlist: skip soft-optimization proposals and spend the budget on ranked pool picks.

    score_main = np.asarray(score_main, dtype=np.float64).reshape(-1)
    ord_main = np.argsort(-np.nan_to_num(score_main, nan=-1e18))
    source_lists['pool_soft'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in ord_main]

    var_score = np.asarray(var_score, dtype=np.float64).reshape(-1)
    ord_var = np.argsort(-np.nan_to_num(var_score, nan=-1e18))
    source_lists['pool_var'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in ord_var]

    if int(X_cand.shape[0]) > 0:
        perm_rand = np.asarray(rng_np.permutation(int(X_cand.shape[0])), dtype=np.int64).reshape(-1)
        source_lists['pool_random'] = [np.asarray(X_cand[idx], dtype=np.float32) for idx in perm_rand]

    if all(len(v) <= 0 for v in source_lists.values()):
        best_idx = int(np.nanargmax(score_main)) if np.isfinite(score_main).any() else int(rng_np.randint(0, int(X_cand.shape[0])))
        x_best = np.asarray(X_cand[best_idx:best_idx + 1], dtype=np.float32)
        return dict(
            X=x_best,
            src_softopt=np.array([0.0], dtype=np.float64),
            src_pool_soft=np.array([1.0], dtype=np.float64),
            src_pool_var=np.array([0.0], dtype=np.float64),
            src_random=np.array([0.0], dtype=np.float64),
        )

    shortlist_x = []
    shortlist_meta = []
    shortlist_by_key = {}
    cursor = {name: 0 for name in source_lists.keys()}

    def _flag_name_for_source(source_name: str) -> str:
        return {
            'softopt': 'softopt',
            'pool_soft': 'pool_soft',
            'pool_var': 'pool_var',
            'pool_random': 'random',
        }[source_name]

    def _take_from_source(source_name: str, quota: int = None):
        quota_eff = None if quota is None else int(max(0, quota))
        if (quota_eff is not None) and (quota_eff <= 0):
            return
        added = 0
        items = source_lists[source_name]
        while cursor[source_name] < len(items):
            x = np.asarray(items[cursor[source_name]], dtype=np.float32)
            cursor[source_name] += 1
            key = _key_from_x(x)
            idx_seen = shortlist_by_key.get(key)
            if idx_seen is None:
                shortlist_by_key[key] = len(shortlist_x)
                shortlist_x.append(x)
                shortlist_meta.append(dict(
                    softopt=False,
                    pool_soft=False,
                    pool_var=False,
                    random=False,
                ))
                idx_seen = len(shortlist_x) - 1
                added += 1
            shortlist_meta[idx_seen][_flag_name_for_source(source_name)] = True
            if (quota_eff is not None) and (added >= quota_eff):
                break

    _take_from_source('softopt', quota_opt)
    _take_from_source('pool_soft', quota_pool_main)
    _take_from_source('pool_var', quota_pool_var)
    _take_from_source('pool_random', quota_pool_random)

    if len(shortlist_x) < target_total:
        for source_name in ('softopt', 'pool_soft', 'pool_var', 'pool_random'):
            if len(shortlist_x) >= target_total:
                break
            _take_from_source(source_name, target_total - len(shortlist_x))

    if len(shortlist_x) <= 0:
        best_idx = int(np.nanargmax(score_main)) if np.isfinite(score_main).any() else int(rng_np.randint(0, int(X_cand.shape[0])))
        shortlist_x = [np.asarray(X_cand[best_idx], dtype=np.float32)]
        shortlist_meta = [dict(softopt=False, pool_soft=True, pool_var=False, random=False)]

    return dict(
        X=np.stack(shortlist_x, axis=0).astype(np.float32),
        src_softopt=np.asarray([float(m['softopt']) for m in shortlist_meta], dtype=np.float64),
        src_pool_soft=np.asarray([float(m['pool_soft']) for m in shortlist_meta], dtype=np.float64),
        src_pool_var=np.asarray([float(m['pool_var']) for m in shortlist_meta], dtype=np.float64),
        src_random=np.asarray([float(m['random']) for m in shortlist_meta], dtype=np.float64),
    )


def _target_forward_update_coeffs_from_state(target_state, x_new: np.ndarray):
    epi_cur = np.asarray(target_state.get('epi_cur', np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    scale = np.maximum(
        np.asarray(
            target_state.get('scale', np.ones((target_state['mean'].shape[1],), dtype=np.float64)),
            dtype=np.float64,
        ),
        1e-8,
    )
    var_obs_target = np.asarray(
        target_state.get('var_obs', np.zeros_like(target_state['mean'], dtype=np.float64)),
        dtype=np.float64,
    )

    if (not bool(target_state.get('can_update', False))) or epi_cur.size <= 0:
        var_epi_post = np.maximum(epi_cur[:, None] * scale[None, :], 1e-8) if epi_cur.size > 0 else np.zeros_like(target_state['mean'], dtype=np.float64)
        var_post = np.asarray(var_epi_post + var_obs_target, dtype=np.float64)
        return dict(
            gain=np.zeros_like(target_state['mean'], dtype=np.float64),
            epi_new=epi_cur.copy(),
            var_post=np.asarray(var_post, dtype=np.float64),
            var_epi_post=np.asarray(var_epi_post, dtype=np.float64),
            denom=float('nan'),
            epi_x=float('nan'),
        )

    x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
    ell = float(target_state['ell'])
    Xtr = np.asarray(target_state['Xtr'], dtype=np.float64)
    chol = np.asarray(target_state['chol'], dtype=np.float64)
    K_TX = np.asarray(target_state['K_TX'], dtype=np.float64)
    lam_obs = float(target_state['lam_obs'])

    k_xX = rbf_kernel_np(x_new, Xtr, ell).reshape(-1)
    v_x = np.linalg.solve(chol, k_xX.reshape(-1, 1)).reshape(-1)
    epi_x = float(np.clip(1.0 - np.sum(v_x * v_x), 0.0, 1.0))
    a_x = np.linalg.solve(chol.T, v_x.reshape(-1, 1)).reshape(-1)

    k_tx = rbf_kernel_np(np.asarray(target_state['X_target'], dtype=np.float64), x_new, ell).reshape(-1)
    cross = k_tx - K_TX @ a_x
    obs_var_x = np.asarray(
        target_state.get('obs_var_x', np.zeros((target_state['mean'].shape[1],), dtype=np.float64)),
        dtype=np.float64,
    ).reshape(1, -1)
    cov_tx = cross[:, None] * scale[None, :]
    denom_mat = np.maximum(scale[None, :] * max(epi_x + lam_obs, 1e-8) + obs_var_x, 1e-8)
    gain = cov_tx / denom_mat
    var_epi_cur = np.maximum(epi_cur[:, None] * scale[None, :], 1e-8)
    var_epi_post = np.maximum(var_epi_cur - (cov_tx * cov_tx) / denom_mat, 1e-8)
    epi_new = np.clip(np.mean(var_epi_post / scale[None, :], axis=1), 0.0, epi_cur)
    var_post = np.asarray(var_epi_post + var_obs_target, dtype=np.float64)
    return dict(
        gain=np.asarray(gain, dtype=np.float64),
        epi_new=np.asarray(epi_new, dtype=np.float64),
        var_post=np.asarray(var_post, dtype=np.float64),
        var_epi_post=np.asarray(var_epi_post, dtype=np.float64),
        denom=float(np.mean(denom_mat)),
        epi_x=float(epi_x),
    )


def _updated_target_forward_mean_from_state(target_state,
                                            update_coeffs,
                                            phi_obs: np.ndarray,
                                            phi_x_mean: np.ndarray):
    mean_cur = np.asarray(target_state['mean'], dtype=np.float64)
    gain = np.asarray(update_coeffs['gain'], dtype=np.float64)
    phi_obs = np.asarray(phi_obs, dtype=np.float64).reshape(-1)
    phi_x_mean = np.asarray(phi_x_mean, dtype=np.float64).reshape(-1)
    residual = phi_obs - phi_x_mean
    mean_post = mean_cur + gain * residual[None, :]
    return np.asarray(mean_post, dtype=np.float64)


def select_random_query_seq_gp(last_sample,
                               forward_net,
                               phi_last_meas_known,
                               phi_last_full_known,
                               gp_state,
                               outcome_model,
                               noise_cfg,
                               reward_oracle,
                               est_indices,
                               true_indices,
                               control_dim,
                               *,
                               rng_q=None):
    if rng_q is None:
        rng_q = np.random
    select_random_query_seq_gp._last_debug = make_empty_gp_sel_debug('random')
    x_cand = rng_q.rand(control_dim).astype(np.float32)
    phi_cand_mean, phi_cand_var = get_forward_mean_var_np(forward_net, x_cand)
    q_samples = np.asarray(gp_state.posterior_samples, dtype=np.float64)
    pred_stats = compute_anchor_soft_scores_from_mean_phi_gp(
        np.asarray(phi_cand_mean, dtype=np.float64).reshape(1, -1),
        phi_last_meas_known,
        q_samples,
        gp_state,
        noise_cfg=noise_cfg,
    )
    pred_pbar = float(np.asarray(pred_stats['pbar'], dtype=np.float64).reshape(-1)[0])
    pred_soft = float(np.asarray(pred_stats['score'], dtype=np.float64).reshape(-1)[0])
    pred_var = float(np.mean(np.asarray(phi_cand_var, dtype=np.float64)))
    phi_cand_true = oracle_phi_from_x(x_cand, outcome_model, noise_cfg)
    x_cand_exec = _executed_x_from_noise(noise_cfg, x_cand)

    r_last = oracle_reward_from_phi(phi_last_full_known[true_indices], reward_oracle, noise_cfg)
    r_cand = oracle_reward_from_phi(phi_cand_true[true_indices], reward_oracle, noise_cfg)
    I = 1 if r_last > r_cand else -1

    select_random_query_seq_gp._last_debug = dict(
        mode='random',
        score=np.nan,
        pbar=pred_pbar,
        soft_score=pred_soft,
        forward_var=pred_var,
        observed_pbar=np.nan,
        observed_soft_score=np.nan,
        target_cur=np.nan,
        target_post=np.nan,
        target_gain=np.nan,
        shortlist_src_soft=np.nan,
        shortlist_src_var=np.nan,
        shortlist_src_random=np.nan,
        shortlist_size=np.nan,
        target_size=np.nan,
    )

    new_query = (I, phi_last_full_known, phi_cand_true)
    new_a = (last_sample, phi_last_full_known[est_indices])
    new_b = (x_cand_exec, phi_cand_true[est_indices])
    return new_query, new_a, new_b, x_cand_exec


def select_active_query_seq_gp(last_sample,
                               phi_last_meas_known,
                               phi_last_full_known,
                               forward_net,
                               outcome_model,
                               gp_state,
                               control_dim,
                               candidate_pool_size,
                               noise_cfg,
                               reward_oracle,
                               est_indices,
                               true_indices,
                               *,
                               mode: str = 'boundary_lookahead',
                               use_oracle_phi_for_U: bool = False,
                               rng_q=None):
    if rng_q is None:
        rng_q = np.random
    mode_l = "" if mode is None else str(mode).lower().strip()
    if mode_l in ('boundary_lookahead_soft_gp', 'boundary_lookahead_soft'):
        mode_l = 'boundary_lookahead_soft_gp'
    elif mode_l in ('boundary_lookahead_jointmi_gp', 'boundary_lookahead_mi_gp', 'boundary_lookahead_jointmi', 'boundary_lookahead_mi'):
        mode_l = 'boundary_lookahead_jointmi_gp'
    elif mode_l in (
        'boundary_lookahead_jointmi_mean_nogate_gp', 'boundary_lookahead_jointmi_meanobs_nogate_gp',
        'boundary_lookahead_mi_mean_nogate_gp', 'boundary_lookahead_mi_meanobs_nogate_gp',
        'boundary_lookahead_mi_mean_nogate_gp', 'boundary_lookahead_mi_meanobs_nogate_gp',
        'boundary_lookahead_mi_mean_nogate', 'boundary_lookahead_mi_meanobs_nogate',
        'boundary_lookahead_jointmi_mean_no_gate_gp', 'boundary_lookahead_jointmi_meanobs_no_gate_gp',
        'boundary_lookahead_mi_mean_no_gate_gp', 'boundary_lookahead_mi_meanobs_no_gate_gp',
    ):
        mode_l = 'boundary_lookahead_jointmi_mean_nogate_gp'
    elif mode_l in (
        'boundary_lookahead', 'boundary-lookahead', 'boundary lookahead', 'bl',
        'boundary_lookahead_jointmi_mean_gp', 'boundary_lookahead_jointmi_meanobs_gp',
        'boundary_lookahead_mi_mean_gp', 'boundary_lookahead_mi_meanobs_gp',
        'boundary_lookahead_mi_mean_gp', 'boundary_lookahead_mi_meanobs_gp',
        'boundary_lookahead_mi_mean', 'boundary_lookahead_mi_meanobs',
    ):
        mode_l = 'boundary_lookahead_jointmi_mean_gp'
    elif mode_l in (
        'bald_fwd_marginal_gp', 'bald_forward_marginal_gp', 'forward_marginal_bald_gp',
        'gp_bald_fwd_marginal', 'gp_bald_forward_marginal',
        'bald_fwd_marginal', 'bald_forward_marginal', 'bald_fm_gp', 'bald_fm',
        'joint_bald_gp', 'bald_joint_gp',
    ):
        mode_l = 'bald_fwd_marginal_gp'
    elif mode_l in ('mutual_information', 'mutual-information', 'mutual information', 'mi', 'bald_gp', 'bald', 'mi_gp', 'gp_bald', 'gp_mi'):
        mode_l = 'bald_gp'
    else:
        raise ValueError(f"Unknown GP active mode '{mode}'.")
    select_active_query_seq_gp._last_debug = make_empty_gp_sel_debug(mode_l)

    q_samples = np.asarray(gp_state.posterior_samples, dtype=np.float64)
    if q_samples.shape[0] < 2:
        return select_random_query_seq_gp(
            last_sample, forward_net, phi_last_meas_known, phi_last_full_known,
            gp_state, outcome_model, noise_cfg, reward_oracle,
            est_indices, true_indices, control_dim,
            rng_q=rng_q,
        )

    X_cand = rng_q.rand(int(candidate_pool_size), int(control_dim)).astype(np.float32)
    if bool(use_oracle_phi_for_U):
        with torch.no_grad():
            phi_full_oracle = outcome_model(torch.tensor(X_cand, dtype=torch.float32)).cpu().numpy()
        phi_cand_mean = np.asarray(phi_full_oracle[:, est_indices], dtype=np.float64)
        var_pool_epi = np.full_like(phi_cand_mean, 1e-8, dtype=np.float64)
        var_pool_total = np.full_like(phi_cand_mean, 1e-8, dtype=np.float64)
        reducible_frac = np.ones((X_cand.shape[0],), dtype=np.float64)
        var_score = np.zeros((X_cand.shape[0],), dtype=np.float64)
    else:
        fwd_pool = get_forward_mean_var_decomposed_np(forward_net, X_cand, noise_cfg=noise_cfg)
        phi_cand_mean = np.asarray(fwd_pool['mean'], dtype=np.float64)
        var_pool_epi = np.asarray(fwd_pool['var_epi'], dtype=np.float64)
        var_pool_total = np.asarray(fwd_pool['var_total'], dtype=np.float64)
        reducible_frac = _reducible_variance_fraction(var_pool_epi, var_pool_total, axis=1)
        var_score = _reducible_variance_score(var_pool_epi, var_pool_total, axis=1)
    soft_stats = compute_anchor_soft_scores_from_mean_phi_gp(
        phi_cand_mean, phi_last_meas_known, q_samples, gp_state,
        noise_cfg=noise_cfg,
    )
    score_soft = np.asarray(soft_stats['score'], dtype=np.float64)
    soft_pbar_all = np.asarray(soft_stats['pbar'], dtype=np.float64)
    score_inv_main = score_soft
    mi_stats = None
    if mode_l in ('boundary_lookahead_jointmi_gp', 'bald_fwd_marginal_gp'):
        mi_stats = compute_cond_mi_anchor_scores_from_candidates_gp(
            X_cand,
            phi_cand_mean,
            phi_last_meas_known,
            forward_net,
            q_samples,
            gp_state,
            rng_np=rng_q,
            n_fwd_samples=int(max(1, JOINT_FWD_SAMPLES)),
            noise_cfg=noise_cfg,
            phi_cand_var_total=var_pool_total,
        )
        score_inv_main = np.asarray(mi_stats['score'], dtype=np.float64)
    elif mode_l in ('boundary_lookahead_jointmi_mean_gp', 'boundary_lookahead_jointmi_mean_nogate_gp'):
        mean_mi_stats = compute_anchor_bald_scores_from_mean_phi_gp(
            phi_cand_mean, phi_last_meas_known, q_samples, gp_state,
            noise_cfg=noise_cfg,
        )
        score_inv_main = np.asarray(mean_mi_stats['score'], dtype=np.float64)

    if mode_l == 'boundary_lookahead_jointmi_mean_nogate_gp':
        score_rank_main = np.asarray(score_inv_main, dtype=np.float64)
    else:
        score_rank_main = np.asarray(score_inv_main, dtype=np.float64) * np.asarray(reducible_frac, dtype=np.float64)

    if mode_l == 'bald_fwd_marginal_gp':
        score_bald_fm = np.asarray(mi_stats['score'], dtype=np.float64)
        pbar_bald_fm = np.asarray(mi_stats['pbar'], dtype=np.float64)
        soft_bald_fm = 4.0 * pbar_bald_fm * (1.0 - pbar_bald_fm)
        if np.isfinite(score_bald_fm).any():
            best_idx = int(np.nanargmax(score_bald_fm))
            score = float(score_bald_fm[best_idx])
        else:
            best_idx = int(rng_q.randint(0, int(X_cand.shape[0])))
            score = float('nan')

        x_sel = np.asarray(X_cand[best_idx], dtype=np.float32)
        phi_cand_true = oracle_phi_from_x(x_sel, outcome_model, noise_cfg)
        x_sel_exec = _executed_x_from_noise(noise_cfg, x_sel)
        r_last_true = oracle_reward_from_phi(phi_last_full_known[true_indices], reward_oracle, noise_cfg)
        r_cand_true = oracle_reward_from_phi(phi_cand_true[true_indices], reward_oracle, noise_cfg)
        I = 1 if r_last_true > r_cand_true else -1

        select_active_query_seq_gp._last_debug = dict(
            mode='bald_fwd_marginal_gp',
            score=float(score),
            pbar=float(pbar_bald_fm[best_idx]) if np.isfinite(pbar_bald_fm[best_idx]) else float('nan'),
            soft_score=float(soft_bald_fm[best_idx]) if np.isfinite(soft_bald_fm[best_idx]) else float('nan'),
            forward_var=float(var_score[best_idx]) if np.isfinite(var_score[best_idx]) else float('nan'),
            observed_pbar=np.nan,
            observed_soft_score=np.nan,
            target_cur=np.nan,
            target_post=np.nan,
            target_gain=np.nan,
            shortlist_src_soft=np.nan,
            shortlist_src_var=np.nan,
            shortlist_src_random=np.nan,
            shortlist_size=np.nan,
            target_size=np.nan,
        )

        new_query = (I, phi_last_full_known, phi_cand_true)
        new_a = (last_sample, phi_last_full_known[est_indices])
        new_b = (x_sel_exec, phi_cand_true[est_indices])
        return new_query, new_a, new_b, x_sel_exec

    if mode_l == 'bald_gp':
        bald_stats = compute_anchor_bald_scores_from_mean_phi_gp(
            phi_cand_mean, phi_last_meas_known, q_samples, gp_state,
            noise_cfg=noise_cfg,
        )
        score_bald = np.asarray(bald_stats['score'], dtype=np.float64)
        if np.isfinite(score_bald).any():
            best_idx = int(np.nanargmax(score_bald))
        else:
            best_idx = int(rng_q.randint(0, int(X_cand.shape[0])))
        x_sel = np.asarray(X_cand[best_idx], dtype=np.float32)
        phi_cand_true = oracle_phi_from_x(x_sel, outcome_model, noise_cfg)
        x_sel_exec = _executed_x_from_noise(noise_cfg, x_sel)
        r_last_true = oracle_reward_from_phi(phi_last_full_known[true_indices], reward_oracle, noise_cfg)
        r_cand_true = oracle_reward_from_phi(phi_cand_true[true_indices], reward_oracle, noise_cfg)
        I = 1 if r_last_true > r_cand_true else -1

        select_active_query_seq_gp._last_debug = dict(
            mode='bald_gp',
            score=float(score_bald[best_idx]) if np.isfinite(score_bald[best_idx]) else float('nan'),
            pbar=float(soft_pbar_all[best_idx]) if np.isfinite(soft_pbar_all[best_idx]) else float('nan'),
            soft_score=float(score_soft[best_idx]) if np.isfinite(score_soft[best_idx]) else float('nan'),
            forward_var=float(var_score[best_idx]) if np.isfinite(var_score[best_idx]) else float('nan'),
            observed_pbar=np.nan,
            observed_soft_score=np.nan,
            target_cur=np.nan,
            target_post=np.nan,
            target_gain=np.nan,
            shortlist_src_soft=np.nan,
            shortlist_src_var=np.nan,
            shortlist_src_random=np.nan,
            shortlist_size=np.nan,
            target_size=np.nan,
        )

        new_query = (I, phi_last_full_known, phi_cand_true)
        new_a = (last_sample, phi_last_full_known[est_indices])
        new_b = (x_sel_exec, phi_cand_true[est_indices])
        return new_query, new_a, new_b, x_sel_exec

    target_rank = np.nan_to_num(score_rank_main, nan=-1e18)
    idx_target = np.argsort(-target_rank)[:int(min(len(target_rank), BOUNDARY_LOOKAHEAD_TARGET_TOPK))]
    X_target = np.asarray(X_cand[idx_target], dtype=np.float32)
    phi_target_mean = np.asarray(phi_cand_mean[idx_target], dtype=np.float64)
    if bool(use_oracle_phi_for_U):
        var_target_cur = np.full_like(phi_target_mean, 1e-8, dtype=np.float64)
        target_state = dict(
            X_target=np.asarray(X_target, dtype=np.float64),
            mean=np.asarray(phi_target_mean, dtype=np.float64),
            var=np.asarray(var_target_cur, dtype=np.float64),
            var_epi=np.asarray(var_target_cur, dtype=np.float64),
            var_obs=np.zeros_like(var_target_cur, dtype=np.float64),
            scale=np.maximum(np.mean(var_target_cur, axis=0), 1e-8),
            epi_cur=np.zeros((phi_target_mean.shape[0],), dtype=np.float64),
            can_update=False,
        )
        PhiT_cur = np.repeat(phi_target_mean[None, :, :], int(max(1, BOUNDARY_LOOKAHEAD_TARGET_FORWARD_SAMPLES)), axis=0)
    else:
        target_state = prepare_target_forward_state(forward_net, X_target, noise_cfg=noise_cfg)
        var_target_cur = np.asarray(target_state['var'], dtype=np.float64)
        PhiT_cur = sample_phi_from_mean_var_np(
            np.asarray(target_state['mean'], dtype=np.float64),
            var_target_cur,
            int(max(1, BOUNDARY_LOOKAHEAD_TARGET_FORWARD_SAMPLES)),
            rng_np=rng_q,
        )
    if mode_l in ('boundary_lookahead_jointmi_gp', 'boundary_lookahead_jointmi_mean_gp', 'boundary_lookahead_jointmi_mean_nogate_gp'):
        U_target_cur = conditional_mi_from_phi_samples_gp(
            q_samples, phi_last_meas_known, PhiT_cur, gp_state,
            noise_cfg=noise_cfg,
        )
    else:
        U_target_cur = conditional_soft_uncertainty_from_phi_samples_gp(
            q_samples, phi_last_meas_known, PhiT_cur, gp_state,
            noise_cfg=noise_cfg,
        )

    if mode_l in ('boundary_lookahead_jointmi_gp', 'boundary_lookahead_jointmi_mean_gp', 'boundary_lookahead_jointmi_mean_nogate_gp'):
        shortlist = build_boundary_lookahead_shortlist_gp_mi(
            X_cand,
            score_rank_main,
            var_score,
            q_samples,
            phi_last_meas_known,
            gp_state,
            control_dim=int(control_dim),
            forward_net=forward_net,
            outcome_model=outcome_model,
            est_indices=est_indices,
            rng_np=rng_q,
        )
        X_short = np.asarray(shortlist['X'], dtype=np.float32)
        short_src_softopt = np.asarray(shortlist['src_softopt'], dtype=np.float64).reshape(-1)
        short_src_pool_soft = np.asarray(shortlist['src_pool_soft'], dtype=np.float64).reshape(-1)
        short_src_pool_var = np.asarray(shortlist['src_pool_var'], dtype=np.float64).reshape(-1)
        short_src_random = np.asarray(shortlist['src_random'], dtype=np.float64).reshape(-1)
    elif mode_l == 'boundary_lookahead_soft_gp':
        shortlist = build_boundary_lookahead_shortlist_gp_soft(
            X_cand,
            score_rank_main,
            var_score,
            q_samples,
            phi_last_meas_known,
            gp_state,
            control_dim=int(control_dim),
            forward_net=forward_net,
            outcome_model=outcome_model,
            est_indices=est_indices,
            rng_np=rng_q,
        )
        X_short = np.asarray(shortlist['X'], dtype=np.float32)
        short_src_softopt = np.asarray(shortlist['src_softopt'], dtype=np.float64).reshape(-1)
        short_src_pool_soft = np.asarray(shortlist['src_pool_soft'], dtype=np.float64).reshape(-1)
        short_src_pool_var = np.asarray(shortlist['src_pool_var'], dtype=np.float64).reshape(-1)
        short_src_random = np.asarray(shortlist['src_random'], dtype=np.float64).reshape(-1)
    else:
        shortlist_rank = np.nan_to_num(score_rank_main, nan=-1e18)
        idx_soft = np.argsort(-shortlist_rank)[:int(min(len(shortlist_rank), BOUNDARY_LOOKAHEAD_SHORTLIST_SOFT))]
        idx_var = np.argsort(-var_score)[:int(min(len(var_score), BOUNDARY_LOOKAHEAD_SHORTLIST_VARIANCE))]
        idx_rand = rng_q.permutation(int(X_cand.shape[0]))[:int(min(int(X_cand.shape[0]), BOUNDARY_LOOKAHEAD_SHORTLIST_RANDOM))]
        idx_short = np.unique(np.concatenate([idx_soft, idx_var, idx_rand], axis=0)).astype(np.int64)
        X_short = np.asarray(X_cand[idx_short], dtype=np.float32)
        idx_soft_set = set(np.asarray(idx_soft, dtype=np.int64).tolist())
        idx_var_set = set(np.asarray(idx_var, dtype=np.int64).tolist())
        idx_rand_set = set(np.asarray(idx_rand, dtype=np.int64).tolist())
        short_src_softopt = np.zeros((idx_short.shape[0],), dtype=np.float64)
        short_src_pool_soft = np.asarray([float(int(i) in idx_soft_set) for i in idx_short], dtype=np.float64)
        short_src_pool_var = np.asarray([float(int(i) in idx_var_set) for i in idx_short], dtype=np.float64)
        short_src_random = np.asarray([float(int(i) in idx_rand_set) for i in idx_short], dtype=np.float64)

    if X_short.ndim == 1:
        X_short = X_short[None, :]

    if bool(use_oracle_phi_for_U):
        with torch.no_grad():
            phi_short_full_oracle = outcome_model(torch.tensor(X_short, dtype=torch.float32)).cpu().numpy()
        phi_short_mean = np.asarray(phi_short_full_oracle[:, est_indices], dtype=np.float64)
        var_short_epi = np.full_like(phi_short_mean, 1e-8, dtype=np.float64)
        var_short_obs = np.zeros_like(phi_short_mean, dtype=np.float64)
        var_short_total = np.full_like(phi_short_mean, 1e-8, dtype=np.float64)
        var_score_short = np.zeros((X_short.shape[0],), dtype=np.float64)
    else:
        fwd_short = get_forward_mean_var_decomposed_np(forward_net, X_short, noise_cfg=noise_cfg)
        phi_short_mean = np.asarray(fwd_short['mean'], dtype=np.float64)
        var_short_epi = np.asarray(fwd_short['var_epi'], dtype=np.float64)
        var_short_obs = np.asarray(fwd_short['var_obs'], dtype=np.float64)
        var_short_total = np.asarray(fwd_short['var_total'], dtype=np.float64)
        var_score_short = _reducible_variance_score(var_short_epi, var_short_total, axis=1)
    soft_short = compute_anchor_soft_scores_from_mean_phi_gp(
        phi_short_mean, phi_last_meas_known, q_samples, gp_state,
        noise_cfg=noise_cfg,
    )
    soft_score_short = np.asarray(soft_short['score'], dtype=np.float64)
    soft_pbar_short = np.asarray(soft_short['pbar'], dtype=np.float64)

    r_anchor_vec = gp_reward_vectorized(q_samples, np.asarray(phi_last_meas_known, dtype=np.float32).reshape(1, -1), gp_state).reshape(-1)

    best_idx = 0
    best_score = -np.inf
    best_expected_post = np.nan

    for c_idx in range(X_short.shape[0]):
        x_c = np.asarray(X_short[c_idx], dtype=np.float32)
        phi_c_mean = np.asarray(phi_short_mean[c_idx], dtype=np.float64)
        obs_var_x = np.asarray(var_short_obs[c_idx], dtype=np.float64).reshape(-1)
        target_state_local = dict(target_state, obs_var_x=obs_var_x)
        target_update = _target_forward_update_coeffs_from_state(target_state_local, x_c)
        var_target_post = np.asarray(target_update['var_post'], dtype=np.float64)

        if bool(use_oracle_phi_for_U) or mode_l in ('boundary_lookahead_jointmi_mean_gp', 'boundary_lookahead_jointmi_mean_nogate_gp'):
            Phi_c_samp = phi_c_mean.reshape(1, 1, -1).astype(np.float64, copy=False)
        else:
            Phi_c_samp = np.asarray(
                sample_phi_from_mean_var_np(
                    phi_c_mean.reshape(1, -1),
                    np.asarray(var_short_total[c_idx], dtype=np.float64).reshape(1, -1),
                    int(max(1, BOUNDARY_LOOKAHEAD_CANDIDATE_FORWARD_SAMPLES)),
                    rng_np=rng_q,
                ),
                dtype=np.float64,
            )
        if Phi_c_samp.ndim == 2:
            Phi_c_samp = Phi_c_samp[None, :, :]

        expected_post = 0.0
        for m in range(Phi_c_samp.shape[0]):
            phi_cm_vec = np.asarray(Phi_c_samp[m, 0], dtype=np.float64).reshape(-1)
            phi_cm = phi_cm_vec.astype(np.float32, copy=False).reshape(1, -1)
            mean_target_post = _updated_target_forward_mean_from_state(
                target_state_local, target_update, phi_cm_vec, phi_c_mean,
            )
            PhiT_post = sample_phi_from_mean_var_np(
                mean_target_post,
                var_target_post,
                int(max(1, BOUNDARY_LOOKAHEAD_TARGET_FORWARD_SAMPLES)),
                rng_np=rng_q,
            )
            r_c_m = gp_reward_vectorized(q_samples, phi_cm, gp_state).reshape(-1)
            beta_s = np.exp(np.clip(q_samples[:, -1], -10.0, 10.0))
            logits_m = beta_s * (r_anchor_vec - r_c_m)
            p_pos_s = _pref_probs_from_logits(
                logits_m,
                beta=beta_s,
                noise_cfg=noise_cfg,
            )
            q_m = float(np.mean(p_pos_s))

            for prob_y, w_raw in ((q_m, p_pos_s), (1.0 - q_m, 1.0 - p_pos_s)):
                w = np.asarray(w_raw, dtype=np.float64)
                wsum = float(np.sum(w))
                if (not np.isfinite(wsum)) or (wsum <= 1e-12):
                    continue
                w = w / wsum
                if mode_l in ('boundary_lookahead_jointmi_gp', 'boundary_lookahead_jointmi_mean_gp', 'boundary_lookahead_jointmi_mean_nogate_gp'):
                    expected_post += (float(prob_y) / float(Phi_c_samp.shape[0])) * conditional_mi_from_phi_samples_gp(
                        q_samples, phi_last_meas_known, PhiT_post, gp_state,
                        noise_cfg=noise_cfg, reward_weights=w,
                    )
                else:
                    expected_post += (float(prob_y) / float(Phi_c_samp.shape[0])) * conditional_soft_uncertainty_from_phi_samples_gp(
                        q_samples, phi_last_meas_known, PhiT_post, gp_state,
                        noise_cfg=noise_cfg, reward_weights=w,
                    )

        score = float(U_target_cur - expected_post)
        if score > best_score:
            best_score = score
            best_idx = c_idx
            best_expected_post = float(expected_post)

    x_sel = np.asarray(X_short[best_idx], dtype=np.float32)
    phi_cand_true = oracle_phi_from_x(x_sel, outcome_model, noise_cfg)
    x_sel_exec = _executed_x_from_noise(noise_cfg, x_sel)
    r_last_true = oracle_reward_from_phi(phi_last_full_known[true_indices], reward_oracle, noise_cfg)
    r_cand_true = oracle_reward_from_phi(phi_cand_true[true_indices], reward_oracle, noise_cfg)
    I = 1 if r_last_true > r_cand_true else -1

    select_active_query_seq_gp._last_debug = dict(
        mode=mode_l,
        score=float(best_score),
        pbar=float(soft_pbar_short[best_idx]),
        soft_score=float(soft_score_short[best_idx]),
        forward_var=float(var_score_short[best_idx]),
        observed_pbar=np.nan,
        observed_soft_score=np.nan,
        target_cur=float(U_target_cur),
        target_post=float(best_expected_post),
        target_gain=float(best_score),
        shortlist_src_softopt=float(short_src_softopt[best_idx]) if best_idx < short_src_softopt.shape[0] else np.nan,
        shortlist_src_soft=float(short_src_pool_soft[best_idx]) if best_idx < short_src_pool_soft.shape[0] else np.nan,
        shortlist_src_var=float(short_src_pool_var[best_idx]) if best_idx < short_src_pool_var.shape[0] else np.nan,
        shortlist_src_random=float(short_src_random[best_idx]) if best_idx < short_src_random.shape[0] else np.nan,
        shortlist_size=float(X_short.shape[0]),
        target_size=float(X_target.shape[0]),
    )

    new_query = (I, phi_last_full_known, phi_cand_true)
    new_a = (last_sample, phi_last_full_known[est_indices])
    new_b = (x_sel_exec, phi_cand_true[est_indices])
    return new_query, new_a, new_b, x_sel_exec


def eval_pref_error_gp(q_samples: np.ndarray,
                       gp_state: PairwiseRFFGPState,
                       PhiA: np.ndarray,
                       PhiB: np.ndarray,
                       y_true: np.ndarray):
    PhiA = np.asarray(PhiA, dtype=np.float32)
    PhiB = np.asarray(PhiB, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)

    beta = np.exp(np.clip(q_samples[:, -1], -10.0, 10.0)).reshape(-1, 1)
    rA = gp_reward_vectorized(q_samples, PhiA, gp_state)
    rB = gp_reward_vectorized(q_samples, PhiB, gp_state)
    logits = beta * (rA - rB)
    p_s = _sigmoid(logits)
    pbar_pos = np.mean(p_s, axis=0)

    y_pred = np.where(pbar_pos >= 0.5, 1, -1)
    err = float(np.mean(y_pred != y_true))

    return err


def _clean_phi_batch_for_policy(outcome_model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if isinstance(outcome_model, DynamicOutcomeOracle):
        return outcome_model.mean_phi_batch(X).astype(np.float32, copy=False)
    dev = _module_device(outcome_model) if isinstance(outcome_model, nn.Module) else torch.device("cpu")
    with torch.no_grad():
        phi = outcome_model(torch.tensor(X, dtype=torch.float32, device=dev))
    return phi.detach().cpu().numpy().astype(np.float32, copy=False)


def _clean_reward_batch_for_policy(reward_oracle, Phi_true_slice: np.ndarray) -> np.ndarray:
    Phi_true_slice = np.asarray(Phi_true_slice, dtype=np.float32)
    if hasattr(reward_oracle, "clean_reward_batch"):
        return np.asarray(reward_oracle.clean_reward_batch(Phi_true_slice), dtype=np.float64).reshape(-1)
    with torch.no_grad():
        rr = reward_oracle(torch.tensor(Phi_true_slice, dtype=torch.float32))
    return rr.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)


def build_policy_eval_pack(outcome_model,
                           reward_oracle,
                           control_dim: int,
                           true_idx,
                           *,
                           n_pool: int,
                           seed: int,
                           batch_size: int = POLICY_EVAL_BATCH_DEFAULT,
                           mc_samples: int = POLICY_MC_FWD_SAMPLES_DEFAULT,
                           mc_variance_key: str = POLICY_MC_VARIANCE_KEY_DEFAULT):
    n_pool = int(max(1, n_pool))
    batch_size = int(max(1, batch_size))
    rs = np.random.RandomState(int(seed))
    X = rs.rand(n_pool, int(control_dim)).astype(np.float32)
    true_idx_arr = np.asarray(true_idx, dtype=np.int64)

    true_reward = np.empty((n_pool,), dtype=np.float64)
    for start in range(0, n_pool, batch_size):
        end = min(start + batch_size, n_pool)
        phi_full = _clean_phi_batch_for_policy(outcome_model, X[start:end])
        true_reward[start:end] = _clean_reward_batch_for_policy(
            reward_oracle,
            phi_full[:, true_idx_arr],
        )

    if np.isfinite(true_reward).any():
        best_idx = int(np.nanargmax(true_reward))
        best_reward = float(true_reward[best_idx])
    else:
        best_idx = -1
        best_reward = float("nan")

    return dict(
        X=X,
        true_reward=true_reward,
        best_idx=best_idx,
        best_reward=best_reward,
        batch_size=int(batch_size),
        mc_samples=int(mc_samples),
        mc_variance_key=str(mc_variance_key),
    )


def _gp_reward_posterior_mean_weights(gp_state: PairwiseRFFGPState) -> np.ndarray:
    q_mean = np.asarray(getattr(gp_state, "map_q", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    feature_dim = int(getattr(getattr(gp_state, "feat", None), "output_dim", max(0, q_mean.size - 1)))
    if q_mean.size == feature_dim + 1 and np.isfinite(q_mean).all():
        return q_mean[:-1].astype(np.float64, copy=False)

    q_samples = np.asarray(getattr(gp_state, "posterior_samples", np.zeros((0, 0))), dtype=np.float64)
    if q_samples.ndim == 2 and q_samples.shape[1] == feature_dim + 1 and q_samples.shape[0] > 0:
        return np.nanmean(q_samples[:, :-1], axis=0).astype(np.float64, copy=False)
    return np.zeros((feature_dim,), dtype=np.float64)


def _gp_reward_posterior_mean_from_phi(phi_batch: np.ndarray,
                                       gp_state: PairwiseRFFGPState,
                                       w_mean: np.ndarray) -> np.ndarray:
    phi_batch = np.asarray(phi_batch, dtype=np.float32)
    if phi_batch.ndim == 1:
        phi_batch = phi_batch[None, :]
    Z = gp_state.feat.transform_np(phi_batch)
    return (Z @ np.asarray(w_mean, dtype=np.float64).reshape(-1)).astype(np.float64, copy=False)


def _policy_variance_key(key) -> str:
    k = str(key).strip().lower()
    if k in ("total", "var_total", "obs", "observation"):
        return "var_total"
    return "var_epi"


def eval_policy_reward_gap_gp(forward_net,
                              gp_state: PairwiseRFFGPState,
                              policy_eval_pack,
                              noise_cfg=None,
                              *,
                              rng_np=None,
                              include_mean_score: bool = False):
    out = dict(
        reward_mc=np.nan,
        regret_mc=np.nan,
        reward_mean=np.nan,
        regret_mean=np.nan,
        reward_mc_minus_mean=np.nan,
        idx_mc=-1,
        idx_mean=-1,
    )
    if policy_eval_pack is None:
        return out
    try:
        n_obs = int(dict(getattr(gp_state, "diag", {})).get("n_obs", 0))
    except Exception:
        n_obs = 0
    if n_obs <= 0:
        return out

    X = np.asarray(policy_eval_pack.get("X", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    true_reward = np.asarray(policy_eval_pack.get("true_reward", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] <= 0 or true_reward.shape[0] != X.shape[0]:
        return out

    best_reward = float(policy_eval_pack.get("best_reward", np.nan))
    if not np.isfinite(best_reward):
        return out

    if rng_np is None:
        rng_np = np.random

    batch_size = int(max(1, policy_eval_pack.get("batch_size", POLICY_EVAL_BATCH_DEFAULT)))
    mc_samples = int(max(1, policy_eval_pack.get("mc_samples", POLICY_MC_FWD_SAMPLES_DEFAULT)))
    var_key = _policy_variance_key(policy_eval_pack.get("mc_variance_key", POLICY_MC_VARIANCE_KEY_DEFAULT))
    w_mean = _gp_reward_posterior_mean_weights(gp_state)
    if w_mean.size <= 0 or not np.isfinite(w_mean).all():
        return out

    n = int(X.shape[0])
    score_mean = np.full((n,), np.nan, dtype=np.float64) if include_mean_score else None
    score_mc = np.full((n,), np.nan, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        fwd = get_forward_mean_var_decomposed_np(forward_net, X[start:end], noise_cfg=noise_cfg)
        phi_mean = np.clip(np.asarray(fwd["mean"], dtype=np.float64), 0.0, 1.0)
        if include_mean_score:
            score_mean[start:end] = _gp_reward_posterior_mean_from_phi(phi_mean, gp_state, w_mean)

        phi_var = np.asarray(fwd[var_key], dtype=np.float64)
        phi_samp = sample_phi_from_mean_var_np(phi_mean, phi_var, mc_samples, rng_np=rng_np)
        m, b, d = phi_samp.shape
        r_flat = _gp_reward_posterior_mean_from_phi(phi_samp.reshape(m * b, d), gp_state, w_mean)
        score_mc[start:end] = np.mean(r_flat.reshape(m, b), axis=0)

    if include_mean_score and score_mean is not None and np.isfinite(score_mean).any():
        idx_mean = int(np.nanargmax(score_mean))
        reward_mean = float(true_reward[idx_mean])
        out["idx_mean"] = idx_mean
        out["reward_mean"] = reward_mean
        out["regret_mean"] = float(best_reward - reward_mean)

    if np.isfinite(score_mc).any():
        idx_mc = int(np.nanargmax(score_mc))
        reward_mc = float(true_reward[idx_mc])
        out["idx_mc"] = idx_mc
        out["reward_mc"] = reward_mc
        out["regret_mc"] = float(best_reward - reward_mc)

    if np.isfinite(out["reward_mc"]) and np.isfinite(out["reward_mean"]):
        out["reward_mc_minus_mean"] = float(out["reward_mc"] - out["reward_mean"])
    return out


def build_nested_phi_pairs_by_tau(
    outcome_model,
    control_dim,
    est_idx,
    true_idx,
    reward_oracle,
    noise_cfg_clean: NoiseConfig,
    tau_list,
    target_kept=TARGET_KEPT,
    pool_size=200_000,
    batch_size=8192,
    # =========================
    # Bin-balanced held-out preference test set
    # =========================
    balanced_test_per_bin: int = 250,
    balanced_test_total: int = None,
    # ==========================================================
    # Extra filter for tau-bin tests and the balanced held-out set:
    # require at least one outcome axis to differ by this much
    # ==========================================================
    min_outcome_axis_gap: float = None,
    control_pair_bank=None,
    axis_gap_idx=None,
):
    """
    Builds reward-gap-bin held-out tests plus one bin-balanced held-out test:
      - phi_pairs_by_tau[tau] contains pairs with abs(reward gap) in [tau, next_tau)
      - phi_pairs_tau_balanced is the bin-balanced held-out test:
          if balanced_test_total is set, split that total across tau bins
          otherwise, from each tau-bin take balanced_test_per_bin
      - if min_outcome_axis_gap is set, all held-out tests
        additionally require max_j |phiA_j - phiB_j| >= threshold on
        axis_gap_idx outcomes. The default is est_idx.

    Returns:
      phi_pairs_by_tau: dict[tau] -> (PhiA, PhiB, y_pm1)
      phi_pairs_tau_balanced: (PhiA, PhiB, y_pm1) or None
      None: kept for compatibility with the closed-loop evaluator signature
    """
    tau_list = [float(t) for t in tau_list]
    tau_list_sorted = sorted(tau_list)
    d_est = len(est_idx)
    gap_idx = est_idx if axis_gap_idx is None else axis_gap_idx
    gap_idx = list(gap_idx)
    min_axis_gap = None if min_outcome_axis_gap is None else float(min_outcome_axis_gap)
    if min_axis_gap is not None:
        if (not np.isfinite(min_axis_gap)) or (min_axis_gap < 0.0):
            raise ValueError("min_outcome_axis_gap must be a finite nonnegative float or None.")

    if control_pair_bank is None:
        XA_bank = None
        XB_bank = None
    else:
        if len(control_pair_bank) != 2:
            raise ValueError("control_pair_bank must be a pair (XA, XB).")
        XA_bank = np.asarray(control_pair_bank[0], dtype=np.float32)
        XB_bank = np.asarray(control_pair_bank[1], dtype=np.float32)
        if XA_bank.ndim != 2 or XB_bank.ndim != 2 or XA_bank.shape != XB_bank.shape:
            raise ValueError("control_pair_bank arrays must both have shape (n_pairs, control_dim).")
        if XA_bank.shape[1] != int(control_dim):
            raise ValueError(f"control_pair_bank control dimension must be {int(control_dim)}.")
        pool_size = int(XA_bank.shape[0])

    is_clean = (noise_cfg_clean.control_jitter_std == 0.0 and
                noise_cfg_clean.oracle_phi_noise_std == 0.0 and
                noise_cfg_clean.reward_noise_std == 0.0)

    # ----------------------------
    # 1) Build a big pool once
    # ----------------------------
    axis_gap_pool = None
    if min_axis_gap is not None and min_axis_gap > 0.0:
        axis_gap_pool = np.empty((pool_size,), dtype=np.float64)

    if not is_clean:
        PhiA_pool = np.zeros((pool_size, d_est), dtype=np.float64)
        PhiB_pool = np.zeros((pool_size, d_est), dtype=np.float64)
        y_pool    = np.zeros((pool_size,), dtype=np.int8)
        absd_pool = np.zeros((pool_size,), dtype=np.float64)

        for i in range(pool_size):
            if XA_bank is None:
                xa, xb = np.random.rand(control_dim), np.random.rand(control_dim)
            else:
                xa, xb = XA_bank[i], XB_bank[i]
            pa_full = oracle_phi_from_x(xa, outcome_model, noise_cfg_clean)
            pb_full = oracle_phi_from_x(xb, outcome_model, noise_cfg_clean)

            diff_true = oracle_reward_from_phi(pa_full[true_idx], reward_oracle, noise_cfg_clean)                         - oracle_reward_from_phi(pb_full[true_idx], reward_oracle, noise_cfg_clean)

            y_pool[i] = 1 if diff_true > 0 else -1
            absd_pool[i] = abs(diff_true)

            PhiA_pool[i, :] = pa_full[est_idx]
            PhiB_pool[i, :] = pb_full[est_idx]
            if axis_gap_pool is not None:
                axis_gap_pool[i] = float(np.max(np.abs(pa_full[gap_idx] - pb_full[gap_idx])))

    else:
        if XA_bank is None:
            XA = np.random.rand(pool_size, control_dim).astype(np.float32)
            XB = np.random.rand(pool_size, control_dim).astype(np.float32)
        else:
            XA = XA_bank
            XB = XB_bank

        PhiA_pool = np.empty((pool_size, d_est), dtype=np.float64)
        PhiB_pool = np.empty((pool_size, d_est), dtype=np.float64)
        absd_pool = np.empty((pool_size,), dtype=np.float64)
        y_pool    = np.empty((pool_size,), dtype=np.int8)

        with torch.no_grad():
            for s in range(0, pool_size, batch_size):
                e = min(pool_size, s + batch_size)
                xa = torch.tensor(XA[s:e], dtype=torch.float32)
                xb = torch.tensor(XB[s:e], dtype=torch.float32)

                phiA_full = outcome_model(xa).cpu().numpy()
                phiB_full = outcome_model(xb).cpu().numpy()

                PhiA_pool[s:e, :] = phiA_full[:, est_idx].astype(np.float64, copy=False)
                PhiB_pool[s:e, :] = phiB_full[:, est_idx].astype(np.float64, copy=False)
                if axis_gap_pool is not None:
                    axis_gap_pool[s:e] = np.max(np.abs(phiA_full[:, gap_idx] - phiB_full[:, gap_idx]), axis=1)

                rA = reward_oracle(torch.tensor(phiA_full[:, true_idx], dtype=torch.float32))
                rB = reward_oracle(torch.tensor(phiB_full[:, true_idx], dtype=torch.float32))
                rA = rA.detach().cpu().numpy().reshape(-1)
                rB = rB.detach().cpu().numpy().reshape(-1)

                diff = (rA - rB).astype(np.float64, copy=False)
                absd_pool[s:e] = np.abs(diff)
                y_pool[s:e] = (diff > 0).astype(np.int8) * 2 - 1

    if min_axis_gap is not None and min_axis_gap > 0.0:
        heldout_mask = (axis_gap_pool >= min_axis_gap)
        heldout_filter_desc = f", max|delta outcome| >= {min_axis_gap:g}"
        n_eligible = int(np.sum(heldout_mask))
        if n_eligible == 0:
            print(
                f"[WARN] outcome-axis filter max|delta outcome| >= {min_axis_gap:g} "
                f"removed all {pool_size} pool pairs."
            )
    else:
        heldout_mask = np.ones((pool_size,), dtype=bool)
        heldout_filter_desc = ""

    # ----------------------------
    # 2) tau-bins: [tau_i, tau_{i+1})
    # ----------------------------
    edges = tau_list_sorted + [float('inf')]
    phi_pairs_by_tau = {}

    for i, tau in enumerate(tau_list_sorted):
        lo = tau
        hi = edges[i+1]

        idx = np.where((absd_pool >= lo) & (absd_pool < hi) & heldout_mask)[0]

        if idx.size == 0:
            print(
                f"[WARN] tau-bin [{lo:g}, {hi if np.isfinite(hi) else 'inf'})"
                f"{heldout_filter_desc}: no pairs in pool_size={pool_size}. Returning empty set."
            )
            phi_pairs_by_tau[tau] = (np.zeros((0, d_est), dtype=np.float64),
                                     np.zeros((0, d_est), dtype=np.float64),
                                     np.zeros((0,), dtype=int))
        else:
            take = min(int(target_kept), int(idx.size))
            if take < target_kept:
                print(
                    f"[WARN] tau-bin [{lo:g}, {hi if np.isfinite(hi) else 'inf'})"
                    f"{heldout_filter_desc}: only {take} pairs available "
                    f"(< target_kept={target_kept})."
                )

            use_idx = np.random.choice(idx, size=take, replace=False) if idx.size > take else idx

            phi_pairs_by_tau[tau] = (
                PhiA_pool[use_idx].copy(),
                PhiB_pool[use_idx].copy(),
                y_pool[use_idx].astype(int).copy()
            )

    # ----------------------------
    # 3) Bin-balanced held-out preference test set
    # ----------------------------
    phi_pairs_tau_balanced = None

    bin_arrays = []
    available_counts = []
    A_list, B_list, y_list = [], [], []
    for tau in tau_list_sorted:
        A, B, y = phi_pairs_by_tau[tau]
        n = int(y.shape[0])
        bin_arrays.append((A, B, y))
        available_counts.append(n)

    if balanced_test_total is not None:
        balanced_counts = _balanced_counts_for_total(available_counts, int(balanced_test_total))
        if int(np.sum(balanced_counts)) < int(balanced_test_total):
            print(
                f"[WARN] bin-balanced held-out test set: only {int(np.sum(balanced_counts))} pairs available "
                f"(< balanced_test_total={int(balanced_test_total)})."
            )
    else:
        per = int(balanced_test_per_bin)
        balanced_counts = np.asarray([min(per, n) for n in available_counts], dtype=np.int64)

    for (A, B, y), take in zip(bin_arrays, balanced_counts):
        n = int(y.shape[0])
        take = int(take)
        if n <= 0 or take <= 0:
            continue
        pick = np.random.choice(n, size=take, replace=False) if n > take else np.arange(n)
        A_list.append(A[pick].copy())
        B_list.append(B[pick].copy())
        y_list.append(y[pick].copy())

    if len(A_list) == 0:
        print("[WARN] bin-balanced held-out test set: all bins empty. Returning empty set.")
        phi_pairs_tau_balanced = (
            np.zeros((0, d_est), dtype=np.float64),
            np.zeros((0, d_est), dtype=np.float64),
            np.zeros((0,), dtype=int)
        )
    else:
        PhiA_bal = np.concatenate(A_list, axis=0)
        PhiB_bal = np.concatenate(B_list, axis=0)
        y_bal = np.concatenate(y_list, axis=0).astype(int, copy=False)
        phi_pairs_tau_balanced = (PhiA_bal, PhiB_bal, y_bal)

    return phi_pairs_by_tau, phi_pairs_tau_balanced, None



def mean_ci(arr, axis=1):
    Tn, R = arr.shape
    df = R - 1
    m = arr.mean(axis=axis)
    if R > 1:
        t95 = t.ppf(0.975, df)
        ci = t95 * arr.std(axis=axis, ddof=1) / np.sqrt(R)
    else:
        ci = np.zeros_like(m)
    return m, ci

def print_matrix_with_header(name, x_ext, mat):
    print(f"\n[{name}] raw (rows = qc in {list(map(int,x_ext))}, cols = runs):")
    with np.printoptions(precision=4, suppress=True, linewidth=200):
        print(mat)


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _inverse_reward_model_type(reward_cfg) -> str:
    model_type = str(_cfg_get(reward_cfg, "model_type", "gp_rff")).lower().strip()
    if model_type in ("gp", "gp_rff", "rff_gp"):
        return "gp_rff"
    if model_type in ("linear", "linear_laplace"):
        return "linear_laplace"
    raise ValueError(
        "learner reward model must be 'gp_rff' or 'linear_laplace' "
        f"(got {model_type!r})."
    )


def compute_metrics_multi_mode(params):
    """
    Run one closed-loop SFIHILO simulation with a GP preference model.

    The return tuple contains only the paper-facing metrics saved by the public runner.
    """
    params_core = params[:30]
    (qc_list, bin_labels, phi_pairs_by_bin, phi_pairs_ge_balanced, phi_pairs_ge_tau,
     init_q, tda, tdb, control_dim, noise_cfg,
     reward_oracle, est_idx, true_idx, n_saved_mcmc, cov0,
     outcome_model, forward_net, seed, total_dim,
     candidate_pool_size, n_test_rmse, reward_cfg,
     n_use_eval, mcmc_sweep_save, mcmc_sample_beta,
     query_policy, active_mode,
     target_err_rate, target_err_trigger, target_err_K) = params_core
    if len(params) >= 32:
        forward_weight_mode = str(params[30]).lower().strip()
        use_oracle_phi_for_U = bool(params[31])
    else:
        forward_weight_mode = "none"
        use_oracle_phi_for_U = False
    if len(params) >= 33:
        policy_eval_pack = params[32]
    else:
        policy_eval_pack = None

    qc_list = sorted(list(qc_list))
    max_q = int(max(qc_list))
    loop_max = max_q
    bin_labels = list(bin_labels)
    K = len(bin_labels)
    d_est = len(est_idx)

    torch.manual_seed(int(seed))
    forward_net.eval()
    outcome_model.eval()

    drift_oracle = get_drift_oracle(outcome_model)
    if drift_oracle is not None:
        drift_oracle.reset_state(seen_x=[np.asarray(tda[0], dtype=np.float64), np.asarray(tdb[0], dtype=np.float64)])
    drift_eval_enabled = bool(drift_oracle is not None and bool(getattr(drift_oracle.cfg, "enabled", True)))
    if drift_eval_enabled:
        drift_pair_bank_size = int(max(1, getattr(drift_oracle.cfg, "eval_pair_bank", 2000)))
        drift_forward_bank_size = int(max(1, getattr(drift_oracle.cfg, "eval_forward_bank", 256)))
        drift_balanced_total = int(max(1, globals().get("GENERIC_BALANCED_TOTAL", globals().get("TARGET_KEPT", 1000))))
        drift_min_axis_gap = float(globals().get("MIN_OUTCOME_AXIS_GAP_TAU_AND_GENERIC_B", 0.0))
        rng_eval = np.random.RandomState(int(seed) + 44000)
        X_eval_forward = rng_eval.rand(drift_forward_bank_size, int(control_dim)).astype(np.float32)
        XA_eval_pairs, XB_eval_pairs = build_control_pair_bank(
            control_dim,
            drift_pair_bank_size,
            seed=int(seed) + 44111,
        )
    else:
        X_eval_forward = None
        XA_eval_pairs = None
        XB_eval_pairs = None
        drift_balanced_total = 0
        drift_min_axis_gap = 0.0

    def _drift_query(offset_queries: int = 0):
        if drift_oracle is None:
            return None
        return float(getattr(drift_oracle, "query_count", 0) + int(offset_queries))

    def _drift_alpha(offset_queries: int = 0):
        if drift_oracle is None:
            return None
        return float(drift_oracle.cfg.alpha_at_query(float(_drift_query(offset_queries=int(offset_queries)))))

    def _train_forward_with_drift_context(retune: bool, target_offset_queries: int = 1):
        if drift_oracle is None:
            train_forward_net(
                forward_net, f_inputs, f_outputs,
                epochs=epochs_for_forward(control_dim), lr=1e-2, wd=1e-4,
                retune=bool(retune),
            )
            return
        if is_forward_tpe_schedule_mode(forward_weight_mode):
            train_forward_net_tpe_schedule(
                forward_net,
                f_inputs,
                f_outputs,
                f_query_counts,
                float(_drift_query(offset_queries=int(target_offset_queries))),
                control_dim=int(control_dim),
                seed=int(seed),
                retune=bool(retune),
                alpha_weight_floor=float(getattr(drift_oracle.cfg, "forward_alpha_weight_floor", 1e-4)),
            )
            return
        if is_forward_oracle_alpha_tpe_mode(forward_weight_mode):
            train_forward_net_oracle_alpha_tpe(
                forward_net,
                f_inputs,
                f_outputs,
                f_alphas,
                _drift_alpha(offset_queries=int(target_offset_queries)),
                control_dim=int(control_dim),
                seed=int(seed),
                retune=bool(retune),
                alpha_weight_floor=float(getattr(drift_oracle.cfg, "forward_alpha_weight_floor", 1e-4)),
            )
            return
        train_forward_net(
            forward_net, f_inputs, f_outputs,
            epochs=epochs_for_forward(control_dim), lr=1e-2, wd=1e-4,
            retune=bool(retune),
            sample_alpha=f_alphas,
            current_alpha=_drift_alpha(offset_queries=int(target_offset_queries)),
            drift_weight_mode=forward_weight_mode,
            alpha_weight_bandwidth=float(getattr(drift_oracle.cfg, "forward_alpha_weight_bandwidth", 0.15)),
            alpha_weight_floor=float(getattr(drift_oracle.cfg, "forward_alpha_weight_floor", 1e-4)),
        )

    rng_q = np.random.RandomState(int(seed) + 11)
    rng_misc = np.random.RandomState(int(seed) + 33)

    enable_extra_pref_stats = bool(ENABLE_EXTRA_PREF_STATS)
    enable_diag_metrics = bool(ENABLE_GP_DIAGNOSTICS)
    enable_shortlist_source_logging = bool(ENABLE_SHORTLIST_SOURCE_LOGGING)
    enable_diversity_metrics = bool(ENABLE_QUERY_DIVERSITY_METRICS)
    enable_target_query_metric = bool(ENABLE_TARGET_QUERY_METRIC)

    qp = ("" if query_policy is None else str(query_policy)).strip().lower()
    am = ("" if active_mode is None else str(active_mode)).strip().lower()
    gp_active_mode = am
    if qp == "active":
        soft_boundary_lookahead_modes = {
            "boundary_lookahead_soft_gp",
            "boundary_lookahead_soft",
        }
        jointmi_boundary_lookahead_modes = {
            "boundary_lookahead_jointmi_gp",
            "boundary_lookahead_mi_gp",
            "boundary_lookahead_jointmi",
            "boundary_lookahead_mi",
        }
        jointmi_mean_boundary_lookahead_modes = {
            "boundary_lookahead",
            "boundary-lookahead",
            "boundary lookahead",
            "bl",
            "boundary_lookahead_jointmi_mean_gp",
            "boundary_lookahead_jointmi_meanobs_gp",
            "boundary_lookahead_mi_mean_gp",
            "boundary_lookahead_mi_meanobs_gp",
            "boundary_lookahead_mi_mean_gp",
            "boundary_lookahead_mi_meanobs_gp",
            "boundary_lookahead_mi_mean",
            "boundary_lookahead_mi_meanobs",
        }
        jointmi_mean_nogate_boundary_lookahead_modes = {
            "boundary_lookahead_jointmi_mean_nogate_gp",
            "boundary_lookahead_jointmi_meanobs_nogate_gp",
            "boundary_lookahead_mi_mean_nogate_gp",
            "boundary_lookahead_mi_meanobs_nogate_gp",
            "boundary_lookahead_mi_mean_nogate_gp",
            "boundary_lookahead_mi_meanobs_nogate_gp",
            "boundary_lookahead_mi_mean_nogate",
            "boundary_lookahead_mi_meanobs_nogate",
            "boundary_lookahead_jointmi_mean_no_gate_gp",
            "boundary_lookahead_jointmi_meanobs_no_gate_gp",
            "boundary_lookahead_mi_mean_no_gate_gp",
            "boundary_lookahead_mi_meanobs_no_gate_gp",
        }
        bald_fwd_marginal_modes = {
            "bald_fwd_marginal_gp",
            "bald_forward_marginal_gp",
            "forward_marginal_bald_gp",
            "gp_bald_fwd_marginal",
            "gp_bald_forward_marginal",
            "bald_fwd_marginal",
            "bald_forward_marginal",
            "bald_fm_gp",
            "bald_fm",
            "joint_bald_gp",
            "bald_joint_gp",
        }
        bald_modes = {
            "mutual_information",
            "mutual-information",
            "mutual information",
            "mi",
            "bald_gp",
            "bald",
            "mi_gp",
            "gp_bald",
            "gp_mi",
        }
        if am in soft_boundary_lookahead_modes:
            gp_active_mode = "boundary_lookahead_soft_gp"
        elif am in jointmi_boundary_lookahead_modes:
            gp_active_mode = "boundary_lookahead_jointmi_gp"
        elif am in jointmi_mean_nogate_boundary_lookahead_modes:
            gp_active_mode = "boundary_lookahead_jointmi_mean_nogate_gp"
        elif am in jointmi_mean_boundary_lookahead_modes:
            gp_active_mode = "boundary_lookahead_jointmi_mean_gp"
        elif am in bald_fwd_marginal_modes:
            gp_active_mode = "bald_fwd_marginal_gp"
        elif am in bald_modes:
            gp_active_mode = "bald_gp"
        else:
            raise ValueError(
                "Supported active modes are 'mutual_information' and "
                "'boundary_lookahead' "
                f"(got active_mode='{active_mode}')."
            )

    def _l2(a, b):
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        return float(np.linalg.norm(a - b))

    def _nn_dist(x_new, X_seen, n_seen):
        if n_seen <= 0:
            return float('nan')
        dif = X_seen[:n_seen] - np.asarray(x_new, dtype=np.float64)[None, :]
        d2 = np.sum(dif * dif, axis=1)
        return float(np.sqrt(np.min(d2)))

    def _safe_nanmean_cols(mat):
        mat = np.asarray(mat, dtype=np.float64)
        out = np.full((mat.shape[1],), np.nan, dtype=np.float64)
        for cc in range(mat.shape[1]):
            col = mat[:, cc]
            if np.isfinite(col).any():
                out[cc] = float(np.nanmean(col))
        return out

    def _thin_q_samples(q_samples):
        q_samples = np.asarray(q_samples, dtype=np.float64)
        if q_samples.ndim != 2 or q_samples.shape[0] <= 0:
            return q_samples
        if n_use_eval is None:
            return q_samples
        use = min(int(n_use_eval), int(q_samples.shape[0]))
        if use >= q_samples.shape[0]:
            return q_samples
        idx = np.linspace(0, q_samples.shape[0] - 1, num=max(1, use), dtype=int)
        return q_samples[idx]

    def _eval_pref_error(PhiA_eval, PhiB_eval, y_eval, q_samples_eval):
        PhiA_eval = np.asarray(PhiA_eval, dtype=np.float32)
        PhiB_eval = np.asarray(PhiB_eval, dtype=np.float32)
        y_eval = np.asarray(y_eval, dtype=np.int64).reshape(-1)
        if PhiA_eval.shape[0] <= 0 or y_eval.shape[0] <= 0:
            return float('nan')
        q_use = _thin_q_samples(q_samples_eval)
        if q_use.ndim != 2 or q_use.shape[0] <= 0:
            return float('nan')
        return eval_pref_error_gp(
            q_use, gp_state, PhiA_eval, PhiB_eval, y_eval,
        )

    def _posterior_predictive_p_and_U(q_samples_eval, PhiA_one, PhiB_one):
        q_use = _thin_q_samples(q_samples_eval)
        if q_use.ndim != 2 or q_use.shape[0] <= 0:
            return float('nan'), float('nan')
        pbar = compute_predictive_pbar_from_phi_array_gp(
            q_use,
            np.asarray(PhiA_one, dtype=np.float32),
            np.asarray(PhiB_one, dtype=np.float32).reshape(1, -1),
            gp_state,
            noise_cfg=noise_cfg,
        )
        p = float(np.asarray(pbar, dtype=np.float64).reshape(-1)[0])
        return p, float(4.0 * p * (1.0 - p))

    def _predicted_query_soft_stats(q_samples_eval, x_sel_one, phi_anchor_one):
        q_use = _thin_q_samples(q_samples_eval)
        if q_use.ndim != 2 or q_use.shape[0] <= 0:
            return float('nan'), float('nan'), float('nan')
        fwd_sel = get_forward_mean_var_decomposed_np(
            forward_net,
            np.asarray(x_sel_one, dtype=np.float32).reshape(1, -1),
            noise_cfg=noise_cfg,
        )
        phi_sel_mean = np.asarray(fwd_sel['mean'], dtype=np.float64)
        var_sel_epi = np.asarray(fwd_sel['var_epi'], dtype=np.float64)
        var_sel_total = np.asarray(fwd_sel['var_total'], dtype=np.float64)
        pred_stats = compute_anchor_soft_scores_from_mean_phi_gp(
            np.asarray(phi_sel_mean, dtype=np.float64),
            np.asarray(phi_anchor_one, dtype=np.float64),
            q_use,
            gp_state,
            noise_cfg=noise_cfg,
        )
        pred_p = float(np.asarray(pred_stats['pbar'], dtype=np.float64).reshape(-1)[0])
        pred_u = float(np.asarray(pred_stats['score'], dtype=np.float64).reshape(-1)[0])
        pred_var = float(np.asarray(_reducible_variance_score(var_sel_epi, var_sel_total, axis=1)).reshape(-1)[0])
        return pred_p, pred_u, pred_var

    def _gp_diag_row():
        if not enable_diag_metrics:
            return [np.nan] * 6
        diag = dict(getattr(gp_state, 'diag', {}))
        clipped_total = float(diag.get('clipped_eigs_H', 0)) + float(diag.get('clipped_eigs', 0))
        return [
            float(diag.get('n_obs', np.nan)),
            float(diag.get('beta_mean', np.nan)),
            float(diag.get('beta_map', np.nan)),
            float(diag.get('eta_map', np.nan)),
            float(diag.get('cov_trace', np.nan)),
            clipped_total,
        ]

    err_test_filt_at = []
    err_ge_at = []
    err_ge_tau_at = []
    drift_err_test_at = []
    drift_err_ge_at = []
    rmse_at = []
    err_train_at = []
    diag_at = []
    dr_mean_at = []
    ll_gain_at = []
    objU_mean_at = []
    dx_last_mean_at = []
    dphi_last_mean_at = []
    dx_nn_mean_at = []
    dphi_nn_mean_at = []
    seldiag_mean_at = []
    policy_reward_mc_at = []
    policy_regret_mc_at = []
    policy_reward_mean_at = []
    policy_regret_mean_at = []

    dr_hist = np.full((loop_max,), np.nan, dtype=np.float64)
    objU_hist = np.full((loop_max,), np.nan, dtype=np.float64)
    dx_last_hist = np.full((loop_max,), np.nan, dtype=np.float64)
    dphi_last_hist = np.full((loop_max,), np.nan, dtype=np.float64)
    dx_nn_hist = np.full((loop_max,), np.nan, dtype=np.float64)
    dphi_nn_hist = np.full((loop_max,), np.nan, dtype=np.float64)
    seldiag_hist = np.full((loop_max, len(GP_SELDIAG_LABELS)), np.nan, dtype=np.float64)

    last_x = tda[0]
    last_phi_full = init_q[1]
    last_phi_meas = tda[1]

    PhiA_tr = np.zeros((loop_max, d_est), dtype=np.float64)
    PhiB_tr = np.zeros((loop_max, d_est), dtype=np.float64)
    y_tr = np.zeros((loop_max,), dtype=np.int8)
    n_q = 0

    def append_query_buffers(I_pm1: int, phiA_full: np.ndarray, phiB_full: np.ndarray):
        nonlocal n_q
        A = np.asarray(phiA_full, dtype=np.float64)[est_idx]
        B = np.asarray(phiB_full, dtype=np.float64)[est_idx]
        PhiA_tr[n_q, :] = A
        PhiB_tr[n_q, :] = B
        y_tr[n_q] = int(I_pm1)
        n_q += 1

    def _append_policy_eval(q_marker: int):
        rng_policy = np.random.RandomState(int(seed) + 55000 + 9973 * int(q_marker))
        pol = eval_policy_reward_gap_gp(
            forward_net,
            gp_state,
            policy_eval_pack,
            noise_cfg=noise_cfg,
            rng_np=rng_policy,
            include_mean_score=False,
        )
        policy_reward_mc_at.append(float(pol["reward_mc"]))
        policy_regret_mc_at.append(float(pol["regret_mc"]))
        policy_reward_mean_at.append(float(pol["reward_mean"]))
        policy_regret_mean_at.append(float(pol["regret_mean"]))

    X_seen = np.zeros((2 * loop_max + 4, int(control_dim)), dtype=np.float64)
    Phi_seen = np.zeros((2 * loop_max + 4, int(d_est)), dtype=np.float64)
    n_seen = 0

    def _register_seen_points(new_points):
        nonlocal n_seen
        dx_list = []
        dphi_list = []
        for (x_np, phi_np) in new_points:
            dx_list.append(_nn_dist(x_np, X_seen, n_seen))
            dphi_list.append(_nn_dist(phi_np, Phi_seen, n_seen))
        dx_arr = np.asarray(dx_list, dtype=np.float64)
        dphi_arr = np.asarray(dphi_list, dtype=np.float64)
        dx_nn = float(np.nanmean(dx_arr)) if np.isfinite(dx_arr).any() else float('nan')
        dphi_nn = float(np.nanmean(dphi_arr)) if np.isfinite(dphi_arr).any() else float('nan')
        for (x_np, phi_np) in new_points:
            X_seen[n_seen, :] = np.asarray(x_np, dtype=np.float64).reshape(-1)
            Phi_seen[n_seen, :] = np.asarray(phi_np, dtype=np.float64).reshape(-1)
            n_seen += 1
        return dx_nn, dphi_nn

    if enable_target_query_metric:
        _fg_target_query = float(max(qc_list))
        if target_err_trigger is None:
            _fg_trigger = 0.20
            _fg_active = False
        else:
            _fg_trigger = float(target_err_trigger)
            if _fg_trigger < 0.0:
                _fg_trigger = float('inf')
                _fg_active = True
            else:
                _fg_active = False
    else:
        _fg_target_query = float(max(qc_list))
        _fg_trigger = float('inf')
        _fg_active = False
    _fg_reached = False
    _fg_consec = 0
    _fg_K = int(target_err_K) if target_err_K else 3
    _fg_target = float(target_err_rate) if target_err_rate else 0.10

    f_inputs = [tda[0], tdb[0]]
    f_outputs = [tda[1], tdb[1]]
    init_alpha = 0.0 if drift_oracle is not None else np.nan
    f_alphas = [init_alpha, init_alpha]
    f_query_counts = [0.0, 0.0]
    _train_forward_with_drift_context(retune=True, target_offset_queries=0)

    if enable_diversity_metrics:
        init_points = [
            (np.asarray(tda[0], dtype=np.float64), np.asarray(tda[1], dtype=np.float64)),
            (np.asarray(tdb[0], dtype=np.float64), np.asarray(tdb[1], dtype=np.float64)),
        ]
        _register_seen_points(init_points)

    inverse_model_type = _inverse_reward_model_type(reward_cfg)
    if inverse_model_type == "linear_laplace":
        gp_cfg = PairwiseRFFGPConfig(
            d_est=d_est,
            m_features=d_est,
            lengthscale=float(INVERSE_GP_RBF_ELL),
            prior_std_w=float(_cfg_get(reward_cfg, "prior_std_lin", INVERSE_GP_PRIOR_STD)),
            eta_mu=float(INVERSE_GP_ETA_MU),
            eta_sig=float(INVERSE_GP_ETA_SIG),
            cov_jitter=float(INVERSE_GP_COV_JITTER),
            map_lbfgs_iters=int(INVERSE_GP_MAP_LBFGS_ITERS),
            posterior_samples=int(INVERSE_GP_POST_SAMPLES),
            seed=int(seed),
            feature_type="linear",
            phi_mean=_cfg_get(reward_cfg, "phi_mean", None),
            phi_std=_cfg_get(reward_cfg, "phi_std", None),
        )
    else:
        gp_cfg = PairwiseRFFGPConfig(
            d_est=d_est,
            m_features=int(INVERSE_GP_RFF_M),
            lengthscale=float(INVERSE_GP_RBF_ELL),
            prior_std_w=float(INVERSE_GP_PRIOR_STD),
            eta_mu=float(INVERSE_GP_ETA_MU),
            eta_sig=float(INVERSE_GP_ETA_SIG),
            cov_jitter=float(INVERSE_GP_COV_JITTER),
            map_lbfgs_iters=int(INVERSE_GP_MAP_LBFGS_ITERS),
            posterior_samples=int(INVERSE_GP_POST_SAMPLES),
            seed=int(seed),
            feature_type="rff",
        )
    gp_state = PairwiseRFFGPState(gp_cfg)
    gp_state = fit_pairwise_rff_gp(
        np.zeros((0, d_est), dtype=np.float32),
        np.zeros((0, d_est), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        gp_state,
        warm_start=False,
        seed=int(seed),
    )

    q_samps0 = np.asarray(gp_state.posterior_samples, dtype=np.float64)
    errs0 = np.empty(K, dtype=np.float64)
    for k, lab in enumerate(bin_labels):
        PhiA, PhiB, y = phi_pairs_by_bin[lab]
        errs0[k] = _eval_pref_error(PhiA, PhiB, y, q_samps0)
    err_test_filt_at.append(errs0)

    if phi_pairs_ge_balanced is not None:
        err_ge0 = _eval_pref_error(
            phi_pairs_ge_balanced[0], phi_pairs_ge_balanced[1], phi_pairs_ge_balanced[2], q_samps0,
        )
    else:
        err_ge0 = float('nan')
    err_ge_at.append(float(err_ge0))

    if phi_pairs_ge_tau is not None:
        err_ge_tau0 = _eval_pref_error(
            phi_pairs_ge_tau[0], phi_pairs_ge_tau[1], phi_pairs_ge_tau[2], q_samps0,
        )
    else:
        err_ge_tau0 = float('nan')
    err_ge_tau_at.append(float(err_ge_tau0))

    if drift_eval_enabled:
        drift_pairs_by_bin_now, drift_pairs_ge_balanced_now = _build_current_drift_eval_sets(
            drift_oracle,
            reward_oracle,
            XA_eval_pairs,
            XB_eval_pairs,
            est_idx,
            true_idx,
            bin_labels,
            target_kept=int(TARGET_KEPT),
            balanced_total=int(drift_balanced_total),
            min_outcome_axis_gap=float(drift_min_axis_gap),
        )
        drift_errs0 = np.empty(K, dtype=np.float64)
        for k, lab in enumerate(bin_labels):
            PhiA, PhiB, y = drift_pairs_by_bin_now[float(lab)]
            drift_errs0[k] = _eval_pref_error(PhiA, PhiB, y, q_samps0)
        if drift_pairs_ge_balanced_now is not None:
            drift_err_ge0 = _eval_pref_error(
                drift_pairs_ge_balanced_now[0],
                drift_pairs_ge_balanced_now[1],
                drift_pairs_ge_balanced_now[2],
                q_samps0,
            )
        else:
            drift_err_ge0 = float('nan')
    else:
        drift_errs0 = np.full((K,), np.nan, dtype=np.float64)
        drift_err_ge0 = float('nan')
    drift_err_test_at.append(drift_errs0)
    drift_err_ge_at.append(float(drift_err_ge0))

    if drift_eval_enabled:
        rmse0 = _forward_rmse_drift_bank(
            forward_net,
            drift_oracle,
            X_eval_forward,
            est_idx,
        )
    else:
        rmse0 = _forward_rmse_seeded(
            forward_net, outcome_model, int(n_test_rmse), int(control_dim), est_idx, noise_cfg,
            seed=int(seed) + 33000,
        )
    rmse_at.append(float(rmse0))
    diag_at.append(_gp_diag_row())

    err_train_at.append(float('nan'))
    dr_mean_at.append(float('nan'))
    objU_mean_at.append(float('nan'))
    ll_gain_at.append(float('nan'))
    dx_last_mean_at.append(float('nan'))
    dphi_last_mean_at.append(float('nan'))
    dx_nn_mean_at.append(float('nan'))
    dphi_nn_mean_at.append(float('nan'))
    seldiag_mean_at.append(np.full((len(GP_SELDIAG_LABELS),), np.nan, dtype=np.float64))
    _append_policy_eval(0)

    for j in range(1, loop_max + 1):
        prev_last_x = np.asarray(last_x, dtype=np.float64)
        prev_last_phi = np.asarray(last_phi_meas, dtype=np.float64)
        q_samps_pre = np.asarray(gp_state.posterior_samples, dtype=np.float64)

        if qp == "random":
            new_q, a_dat, b_dat, last_x = select_random_query_seq_gp(
                last_x, forward_net, last_phi_meas, last_phi_full,
                gp_state, outcome_model, noise_cfg, reward_oracle,
                est_idx, true_idx, control_dim,
                rng_q=rng_q,
            )
        elif qp == "active":
            new_q, a_dat, b_dat, last_x = select_active_query_seq_gp(
                last_x, last_phi_meas, last_phi_full,
                forward_net, outcome_model, gp_state,
                control_dim, candidate_pool_size,
                noise_cfg, reward_oracle,
                est_idx, true_idx,
                mode=gp_active_mode,
                use_oracle_phi_for_U=bool(use_oracle_phi_for_U),
                rng_q=rng_q,
            )
        else:
            raise ValueError(f"Unknown query_policy='{query_policy}' (expected 'random' or 'active').")

        if qp == "active":
            sel_dbg = dict(getattr(select_active_query_seq_gp, "_last_debug", make_empty_gp_sel_debug(gp_active_mode)))
        else:
            sel_dbg = dict(getattr(select_random_query_seq_gp, "_last_debug", make_empty_gp_sel_debug("random")))

        dx_last = float('nan')
        dphi_last = float('nan')
        dx_nn = float('nan')
        dphi_nn = float('nan')
        if enable_diversity_metrics:
            new_points = [(np.asarray(b_dat[0], dtype=np.float64), np.asarray(b_dat[1], dtype=np.float64))]
            dx_last = _l2(np.asarray(b_dat[0], dtype=np.float64), prev_last_x)
            dphi_last = _l2(np.asarray(b_dat[1], dtype=np.float64), prev_last_phi)
            dx_nn, dphi_nn = _register_seen_points(new_points)
            dx_last_hist[j - 1] = float(dx_last)
            dphi_last_hist[j - 1] = float(dphi_last)
            dx_nn_hist[j - 1] = float(dx_nn)
            dphi_nn_hist[j - 1] = float(dphi_nn)

        abs_dr = float('nan')
        p_query = float('nan')
        U_query = float('nan')
        need_query_soft_stats = bool(enable_extra_pref_stats or enable_diag_metrics or enable_shortlist_source_logging)
        if need_query_soft_stats:
            PhiA_est_obs = np.asarray(new_q[1], dtype=np.float64)[est_idx]
            PhiB_est_obs = np.asarray(new_q[2], dtype=np.float64)[est_idx]
            p_query, U_query = _posterior_predictive_p_and_U(q_samps_pre, PhiA_est_obs, PhiB_est_obs)
        if enable_extra_pref_stats:
            rA_true = oracle_reward_from_phi_preserve_noise_state(
                np.asarray(new_q[1], dtype=np.float32)[true_idx], reward_oracle, noise_cfg
            )
            rB_true = oracle_reward_from_phi_preserve_noise_state(
                np.asarray(new_q[2], dtype=np.float32)[true_idx], reward_oracle, noise_cfg
            )
            abs_dr = float(abs(rA_true - rB_true))
        if enable_diag_metrics or enable_shortlist_source_logging:
            need_pred_fill = (
                (not np.isfinite(float(sel_dbg.get('pbar', np.nan))))
                or (not np.isfinite(float(sel_dbg.get('soft_score', np.nan))))
                or (not np.isfinite(float(sel_dbg.get('forward_var', np.nan))))
            )
            if need_pred_fill:
                pred_pbar, pred_soft, pred_var = _predicted_query_soft_stats(
                    q_samps_pre,
                    np.asarray(b_dat[0], dtype=np.float32),
                    last_phi_meas,
                )
                if not np.isfinite(float(sel_dbg.get('pbar', np.nan))):
                    sel_dbg['pbar'] = float(pred_pbar)
                if not np.isfinite(float(sel_dbg.get('soft_score', np.nan))):
                    sel_dbg['soft_score'] = float(pred_soft)
                if not np.isfinite(float(sel_dbg.get('forward_var', np.nan))):
                    sel_dbg['forward_var'] = float(pred_var)
            if np.isfinite(p_query):
                sel_dbg['observed_pbar'] = float(p_query)
            if np.isfinite(U_query):
                sel_dbg['observed_soft_score'] = float(U_query)
            seldiag_hist[j - 1, :] = gp_sel_debug_to_array(sel_dbg)

        append_query_buffers(new_q[0], new_q[1], new_q[2])
        if enable_extra_pref_stats:
            dr_hist[n_q - 1] = abs_dr
            objU_hist[n_q - 1] = float(U_query)

        f_inputs.append(b_dat[0])
        f_outputs.append(b_dat[1])
        if drift_oracle is not None:
            f_alphas.append(_drift_alpha(offset_queries=0))
            f_query_counts.append(float(_drift_query(offset_queries=0)))
        _train_forward_with_drift_context(retune=(j in qc_list), target_offset_queries=1)

        gp_state = fit_pairwise_rff_gp(
            np.asarray(PhiA_tr[:n_q], dtype=np.float32),
            np.asarray(PhiB_tr[:n_q], dtype=np.float32),
            np.asarray(y_tr[:n_q], dtype=np.int64),
            gp_state,
            warm_start=True,
            seed=int(seed) + j,
        )

        if drift_oracle is not None:
            x_exec = getattr(drift_oracle, "last_observed_x", None)
            x_load = np.asarray(b_dat[0] if x_exec is None else x_exec, dtype=np.float64)
            drift_oracle.advance_after_query(x_load)

        last_phi_full = new_q[2]
        last_phi_meas = b_dat[1]

        if enable_target_query_metric and _fg_active and (not _fg_reached) and (phi_pairs_ge_tau is not None):
            q_samps_fg = np.asarray(gp_state.posterior_samples, dtype=np.float64)
            err_ge_fg = _eval_pref_error(
                phi_pairs_ge_tau[0], phi_pairs_ge_tau[1], phi_pairs_ge_tau[2], q_samps_fg,
            )
            if np.isfinite(err_ge_fg) and (err_ge_fg <= _fg_target):
                _fg_consec += 1
                if _fg_consec >= _fg_K:
                    _fg_target_query = float(j - _fg_K + 1)
                    _fg_reached = True
            else:
                _fg_consec = 0

        if j in qc_list:
            q_samps = np.asarray(gp_state.posterior_samples, dtype=np.float64)

            errs = np.empty(K, dtype=np.float64)
            for k, lab in enumerate(bin_labels):
                PhiA, PhiB, y = phi_pairs_by_bin[lab]
                errs[k] = _eval_pref_error(PhiA, PhiB, y, q_samps)
            err_test_filt_at.append(errs)

            if phi_pairs_ge_balanced is not None:
                err_ge = _eval_pref_error(
                    phi_pairs_ge_balanced[0], phi_pairs_ge_balanced[1], phi_pairs_ge_balanced[2], q_samps,
                )
            else:
                err_ge = float('nan')
            err_ge_at.append(float(err_ge))

            if phi_pairs_ge_tau is not None:
                err_ge_tau = _eval_pref_error(
                    phi_pairs_ge_tau[0], phi_pairs_ge_tau[1], phi_pairs_ge_tau[2], q_samps,
                )
            else:
                err_ge_tau = float('nan')
            err_ge_tau_at.append(float(err_ge_tau))

            if drift_eval_enabled:
                drift_pairs_by_bin_now, drift_pairs_ge_balanced_now = _build_current_drift_eval_sets(
                    drift_oracle,
                    reward_oracle,
                    XA_eval_pairs,
                    XB_eval_pairs,
                    est_idx,
                    true_idx,
                    bin_labels,
                    target_kept=int(TARGET_KEPT),
                    balanced_total=int(drift_balanced_total),
                    min_outcome_axis_gap=float(drift_min_axis_gap),
                )
                drift_errs = np.empty(K, dtype=np.float64)
                for k, lab in enumerate(bin_labels):
                    PhiA, PhiB, y = drift_pairs_by_bin_now[float(lab)]
                    drift_errs[k] = _eval_pref_error(PhiA, PhiB, y, q_samps)
                if drift_pairs_ge_balanced_now is not None:
                    drift_err_ge = _eval_pref_error(
                        drift_pairs_ge_balanced_now[0],
                        drift_pairs_ge_balanced_now[1],
                        drift_pairs_ge_balanced_now[2],
                        q_samps,
                    )
                else:
                    drift_err_ge = float('nan')
            else:
                drift_errs = np.full((K,), np.nan, dtype=np.float64)
                drift_err_ge = float('nan')
            drift_err_test_at.append(drift_errs)
            drift_err_ge_at.append(float(drift_err_ge))

            if enable_target_query_metric and (not _fg_active) and (not _fg_reached) and np.isfinite(err_ge_tau) and (err_ge_tau <= _fg_trigger):
                _fg_active = True

            if n_q > 0:
                tr_err = _eval_pref_error(PhiA_tr[:n_q], PhiB_tr[:n_q], y_tr[:n_q].astype(int), q_samps)
            else:
                tr_err = float('nan')
            err_train_at.append(float(tr_err))

            if enable_extra_pref_stats:
                ll_gain = float('nan')
                dr_slice = np.asarray(dr_hist[:n_q], dtype=np.float64)
                obj_slice = np.asarray(objU_hist[:n_q], dtype=np.float64)
                dr_mean = float(np.nanmean(dr_slice)) if np.isfinite(dr_slice).any() else float('nan')
                objU_mean = float(np.nanmean(obj_slice)) if np.isfinite(obj_slice).any() else float('nan')
            else:
                ll_gain = float('nan')
                dr_mean = float('nan')
                objU_mean = float('nan')
            ll_gain_at.append(float(ll_gain))
            dr_mean_at.append(float(dr_mean))
            objU_mean_at.append(float(objU_mean))

            if drift_eval_enabled:
                rmse_here = _forward_rmse_drift_bank(
                    forward_net,
                    drift_oracle,
                    X_eval_forward,
                    est_idx,
                )
            else:
                rmse_here = _forward_rmse_seeded(
                    forward_net, outcome_model, int(n_test_rmse), int(control_dim), est_idx, noise_cfg,
                    seed=int(seed) + 33000 + j,
                )
            rmse_at.append(float(rmse_here))
            diag_at.append(_gp_diag_row())

            if enable_diversity_metrics:
                dx_last_mean_at.append(float(np.nanmean(dx_last_hist[:j])))
                dphi_last_mean_at.append(float(np.nanmean(dphi_last_hist[:j])))
                dx_nn_mean_at.append(float(np.nanmean(dx_nn_hist[:j])))
                dphi_nn_mean_at.append(float(np.nanmean(dphi_nn_hist[:j])))
            else:
                dx_last_mean_at.append(float('nan'))
                dphi_last_mean_at.append(float('nan'))
                dx_nn_mean_at.append(float('nan'))
                dphi_nn_mean_at.append(float('nan'))

            if enable_diag_metrics or enable_shortlist_source_logging:
                seldiag_mean_at.append(_safe_nanmean_cols(seldiag_hist[:j, :]))
            else:
                seldiag_mean_at.append(np.full((len(GP_SELDIAG_LABELS),), np.nan, dtype=np.float64))
            _append_policy_eval(j)

    return (
        np.vstack(err_test_filt_at),
        np.asarray(err_ge_at, dtype=np.float64),
        np.asarray(rmse_at, dtype=np.float64),
        np.asarray(policy_reward_mc_at, dtype=np.float64),
        np.asarray(policy_regret_mc_at, dtype=np.float64),
        np.asarray(policy_reward_mean_at, dtype=np.float64),
        np.asarray(policy_regret_mean_at, dtype=np.float64),
        np.vstack(drift_err_test_at),
        np.asarray(drift_err_ge_at, dtype=np.float64),
    )

