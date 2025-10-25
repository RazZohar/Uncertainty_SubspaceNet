
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .system_model import SystemModelParams
from .signal_creation import *
from .data_handler import *
from .criterions import set_criterions
from .training import *
from .evaluation import evaluate
from .plotting import initialize_figures
from pathlib import Path
from .models import ModelGenerator

from .methods import MUSIC, RootMUSIC, Esprit, MVDR

#from .create_codebook import create_codebook as codebook_creation

#import .qunatizer as quantizer

import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np


# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

# Use this flag to generate graph
plot_spectrum_flag = True


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
from typing import Optional

from src.criterions import MSPE_Empricial, BPE
# =========================== Utilities ===========================

C0 = 299_792_458.0  # m/s

def wavelength_from_fc(fc_hz, vp=C0, n=None):
    if n is not None:
        vp = C0 / n
    return vp / fc_hz

def estimate_covariance_matrix(Z: np.ndarray) -> np.ndarray:
    return (Z @ Z.conj().T) / Z.shape[1]

def shrinkage_cov(Rhat: np.ndarray, rho: float = 0.0) -> np.ndarray:
    """Optional diagonal loading. Set rho=0.0 to disable."""
    if rho <= 0.0:
        return Rhat
    M = Rhat.shape[0]
    mu = np.trace(Rhat) / M
    return (1 - rho) * Rhat + rho * mu * np.eye(M, dtype=complex)

# ===================== ESPRIT (overlapped, Δ=λ/2) =====================

def build_overlapped_selectors(M: int, shift: int = 1):
    rows = M - shift
    J1 = np.zeros((rows, M)); J2 = np.zeros((rows, M))
    for r in range(rows):
        J1[r, r] = 1.0
        J2[r, r + shift] = 1.0
    return J1, J2

def esprit_overlapped(Rhat: np.ndarray, d_sources: int, shift: int = 1):
    """
    Classical ESPRIT with overlapped subarrays:
      E_x = J1 E_s,  E_y = J2 E_s,  F = pinv(E_x) @ E_y
    Returns right/left eigenvectors of F and selectors J1,J2.
    """
    M = Rhat.shape[0]
    evals, evecs = np.linalg.eigh(Rhat)
    idx = np.argsort(evals)[::-1][:d_sources].copy()
    E_s = evecs[:, idx]
    J1, J2 = build_overlapped_selectors(M, shift=shift)
    E_x, E_y = J1 @ E_s, J2 @ E_s
    F = np.linalg.pinv(E_x) @ E_y
    lam, V = np.linalg.eig(F)      # right eigvecs (columns)
    Q = np.linalg.inv(V)           # left eigvecs (rows)
    return {"E_s": E_s, "E_x": E_x, "E_y": E_y, "J1": J1, "J2": J2, "F": F,
            "lambda": lam, "V": V, "Q": Q}

# ================== 4th‑order moment (Eq. 63, Gaussian) ==================

def rcov_conj_gaussian_plugin(Rhat: np.ndarray, Ns: int) -> np.ndarray:
    """
    E{ΔR[a1,a2] ΔR[b1,b2]^*} ≈ (1/Ns) * Rhat[a1,b2] * Rhat[b1,a2].
    Shape: (a1,a2,b1,b2)
    """
    return (Rhat[:, None, :, None] * Rhat[None, :, None, :]) / Ns

# =============== Eq. 66: eigenvector perturbation blocks ===============

def compute_delta_s_covariance_blocks_eq66(
    S_full: np.ndarray,
    alpha: np.ndarray,
    Rcov_conj: np.ndarray,
    signal_indices,
    denom_eps: float = 1e-6
):
    """
    Returns covHs_gh, covTs_gh of shape (d, d, M, M), using the "swap" note for the
    unconjugated 4th moment and small denominator regularization denom_eps.
    """
    S_full = np.asarray(S_full); alpha = np.asarray(alpha); Rcov_conj = np.asarray(Rcov_conj)
    M = S_full.shape[0]
    sig_idx = np.asarray(signal_indices); d = sig_idx.size
    Rcov_unconj = np.swapaxes(Rcov_conj, 2, 3)

    covHs_gh = np.zeros((d, d, M, M), dtype=complex)
    covTs_gh = np.zeros((d, d, M, M), dtype=complex)

    for gi, g in enumerate(sig_idx):
        s_g = S_full[:, g]
        for hi, h in enumerate(sig_idx):
            s_h = S_full[:, h]
            accum_H = np.zeros((M, M), dtype=complex)
            accum_T = np.zeros((M, M), dtype=complex)

            for i in range(M):
                if i == g:
                    continue
                s_i = S_full[:, i]; s_i_conj = s_i.conj()
                T1_H = np.tensordot(s_i_conj, Rcov_conj,   axes=(0, 0))  # (a2,b1,b2)
                T1_T = np.tensordot(s_i_conj, Rcov_unconj, axes=(0, 0))  # (a2,b2,b1)
                for n in range(M):
                    if n == h:
                        continue
                    s_n = S_full[:, n]; s_n_conj = s_n.conj()
                    T2_H = np.tensordot(s_g, T1_H, axes=(0, 0))  # (b1,b2)
                    T2_T = np.tensordot(s_g, T1_T, axes=(0, 0))  # (b2,b1)
                    T3_H = np.tensordot(s_n_conj, T2_H, axes=(0, 0))     # (b2)
                    T3_T = np.tensordot(s_n_conj, T2_T, axes=(0, 1))     # (b2)
                    coeff_H = np.dot(s_h, T3_H)
                    coeff_T = np.dot(s_h, T3_T)
                    denom = (alpha[g] - alpha[i]) * (alpha[h] - alpha[n]) + denom_eps
                    accum_H += (coeff_H / denom) * (s_i[:, None] @ s_n_conj[None, :])
                    accum_T += (coeff_T / denom) * (s_i[:, None] @ s_n[None, :])

            # Hermitian symmetrize the ^H block
            accum_H = (accum_H + accum_H.conj().T) / 2
            covHs_gh[gi, hi] = accum_H
            covTs_gh[gi, hi] = accum_T

    return covHs_gh, covTs_gh

# ================= Eq. 52 / Eq. 53 with v‑weighting =================

def _assemble_middle_from_v(v_i: np.ndarray, cov_gh: np.ndarray, conjugate_h: bool = True) -> np.ndarray:
    """
    Build Σ_{g,h} v_{ig} (v_{ih}^* or v_{ih}) * Cov[g,h], Cov[g,h] is (M x M).
    Accepts cov_gh shaped (d,d,M,M) or (M,M,d,d).
    """
    v = np.asarray(v_i).reshape(-1)
    cov = np.asarray(cov_gh)
    if cov.ndim != 4:
        raise ValueError("cov_gh must be 4D.")
    if cov.shape[:2] != (v.size, v.size) and cov.shape[-2:] != (v.size, v.size):
        raise ValueError("cov_gh dims must match len(v_i).")
    if cov.shape[-2:] == (v.size, v.size):  # (M,M,d,d) -> (d,d,M,M)
        cov = np.moveaxis(cov, (-2, -1), (0, 1))
    W = np.outer(v, np.conj(v)) if conjugate_h else np.outer(v, v)
    return np.tensordot(W, cov, axes=([0, 1], [0, 1]))  # (M x M)

def compute_eq52_weighted(lam_i, q_i, E_x, W1, W2, covHs_gh, v_i):
    E_x_pinv = np.linalg.pinv(E_x)
    middle = _assemble_middle_from_v(v_i, covHs_gh, conjugate_h=True)
    left   = q_i @ E_x_pinv @ (W1 - np.conj(lam_i) * W2)                 # (1 x M)
    right  = (W1 - np.conj(lam_i) * W2).conj().T @ E_x_pinv.conj().T @ q_i.conj().T  # (M x 1)
    return (left @ middle @ right).item()

def compute_eq53_weighted(lam_i, q_i, E_x, W1, W2, covTs_gh, v_i):
    E_x_pinv = np.linalg.pinv(E_x)
    middle = _assemble_middle_from_v(v_i, covTs_gh, conjugate_h=False)
    left   = q_i @ E_x_pinv @ (W2 - lam_i * W1)
    right  = (W2 - lam_i * W1).T @ E_x_pinv.T @ q_i.T
    return (left @ middle @ right).item()

# ================= Eq. 58 with Δ = λ/2 simplified =================

def eq58_scale_half_lambda(theta_i: float, eps: float = 1e-8) -> float:
    c2 = max(np.cos(theta_i)**2, eps)  # guard endfire
    return 1.0 / (2 * np.pi**2 * c2)

def compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_val, eq53_val, theta_i,
                                            clip_nonneg: bool = True, eps: float = 1e-8) -> float:
    scale = eq58_scale_half_lambda(theta_i, eps)
    val = scale * (np.real(eq52_val) - np.real(eq53_val * (np.conj(lam_i)**2)))
    return float(max(val, 0.0)) if clip_nonneg else float(val)

# ===================== Data generation (multi‑source) =====================

def generate_signals_multi(doas_deg, M, N, fc_hz, snr_db=-10.0, vp=C0):
    """
    Multiple narrowband sources on a ULA with d = λ/2 spacing.
    """
    doas_rad = np.deg2rad(np.asarray(doas_deg))
    lam = wavelength_from_fc(fc_hz, vp=vp)
    d = lam / 2.0
    k = 2 * np.pi / lam
    n_idx = np.arange(M)[:, None]
    A = np.exp(1j * k * d * n_idx * np.cos(doas_rad))  # (M x d)
    dsrc = len(doas_rad)

    S = (np.random.randn(dsrc, N) + 1j*np.random.randn(dsrc, N)) / np.sqrt(2)  # unit power
    snr_lin = 10**(snr_db/10.0)
    noise_var = 1.0 / snr_lin
    W = np.sqrt(noise_var/2.0) * (np.random.randn(M, N) + 1j*np.random.randn(M, N))
    return A @ S + W


def generate_signals_multi_cohernt(doas_deg, M, N, fc_hz, snr_db=-10.0, vp=C0):
    """
    Multiple narrowband sources on a ULA with d = λ/2 spacing.
    """
    doas_rad = np.deg2rad(np.asarray(doas_deg))
    lam = wavelength_from_fc(fc_hz, vp=vp)
    d = lam / 2.0
    k = 2 * np.pi / lam
    n_idx = np.arange(M)[:, None]
    A = np.exp(1j * k * d * n_idx * np.cos(doas_rad))  # (M x d)
    dsrc = len(doas_rad)

    sig = (

            np.sqrt(2)
            * (
                    np.random.randn(1, N)
                    + 1j * np.random.randn(1, N)
            )
    )
    S = np.repeat(sig, dsrc, axis=0)

    snr_lin = 10**(snr_db/10.0)
    noise_var = 1.0 / snr_lin
    W = np.sqrt(noise_var/2.0) * (np.random.randn(M, N) + 1j*np.random.randn(M, N))
    return A @ S + W

# ===================== Mode‑wise 180° wrapping (π‑periodic) =====================

def wrap_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray, period: float = 180.0) -> np.ndarray:
    """
    Elementwise wrapped difference a-b to (-period/2, period/2].
    For ULA without FB averaging, period = 180°.
    """
    diff = a_deg - b_deg
    return ((diff + period/2) % period) - period/2

# ===================== PyTorch: wrapped empirical covariance =====================

def _wrap_circular_torch(x: torch.Tensor, period: float) -> torch.Tensor:
    """Wrap to (-period/2, period/2]."""
    return ((x + period / 2) % period) - period / 2

def empirical_error_cov(
    theta_hat: torch.Tensor,
    theta_true: torch.Tensor,
    *,
    period: Optional[float] = 180.0,     # 180.0 for degrees (π‑periodic), math.pi for radians, or None
    unbiased: bool = True,
    return_scalar_if_K1: bool = True,
) -> torch.Tensor:
    """
    Empirical covariance of the estimation error e = θ̂ − θ with optional circular wrapping.
    θ tensors shape: (..., K); leading '...' is the number of experiments (N).
    Returns (K,K) covariance in the same units squared as inputs.
    """
    e = theta_hat - theta_true
    if period is not None:
        e = _wrap_circular_torch(e, float(period))
    K = e.shape[-1]
    e = e.reshape(-1, K)                      # (N, K)
    N = e.shape[0]
    if N < 2:
        raise ValueError(f"Need at least 2 samples to compute covariance; got N={N}.")
    e_centered = e - e.mean(dim=0, keepdim=True)
    denom = (N - 1) if unbiased else N
    Sigma = (e_centered.conj().T @ e_centered) / denom
    return Sigma.squeeze() if (return_scalar_if_K1 and K == 1) else Sigma


#=========================================================================================
import numpy as np
from typing import Tuple

# ----------------- Wrapping -----------------
def wrap_diff_deg(a_deg: np.ndarray, b_deg: np.ndarray, period: float = 180.0) -> np.ndarray:
    """
    Elementwise wrapped difference a-b to (-period/2, period/2].
    For a ULA without FB averaging, use period=180° (π-periodic).
    """
    diff = a_deg - b_deg
    return ((diff + period/2) % period) - period/2


# ----------------- Best alignment (perm) per trial -----------------
def _align_hungarian(doa_hat_deg: np.ndarray, doa_true_deg: np.ndarray, period: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hungarian alignment (scales to larger K). Returns:
      perm_truth_to_est: (K,) index of the chosen estimate for each true mode j
      err_deg_aligned:   (K,) wrapped signed errors in truth order
      doa_hat_aligned:   (K,) estimates reordered to truth order
    """
    from scipy.optimize import linear_sum_assignment  # optional dependency

    K = doa_true_deg.size
    # Build KxK cost matrix of squared wrapped errors (deg²)
    C = np.zeros((K, K))
    for i_est in range(K):
        for j_true in range(K):
            e = wrap_diff_deg(doa_hat_deg[i_est], doa_true_deg[j_true], period=period)
            C[i_est, j_true] = e * e  # deg²

    row_ind, col_ind = linear_sum_assignment(C)
    # Build mapping truth j -> chosen est i
    perm_truth_to_est = np.empty(K, dtype=int)
    for i_est, j_true in zip(row_ind, col_ind):
        perm_truth_to_est[j_true] = i_est

    doa_hat_aligned = doa_hat_deg[perm_truth_to_est]  # in truth order
    err_deg_aligned = wrap_diff_deg(doa_hat_aligned, doa_true_deg, period=period)
    return perm_truth_to_est, err_deg_aligned, doa_hat_aligned


def _align_bruteforce(doa_hat_deg: np.ndarray, doa_true_deg: np.ndarray, period: float
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Brute-force alignment (OK for small K, e.g., K<=6).
    """
    from itertools import permutations
    K = doa_true_deg.size
    best_cost = np.inf
    best_perm = None
    best_err = None

    for p in permutations(range(K)):
        e = wrap_diff_deg(doa_hat_deg[list(p)], doa_true_deg, period=period)
        cost = np.sum(e**2)
        if cost < best_cost:
            best_cost = cost
            best_perm = np.array(p, dtype=int)
            best_err = e

    return best_perm, best_err, doa_hat_deg[best_perm]


def align_one_trial(doa_hat_deg: np.ndarray, doa_true_deg: np.ndarray, period: float = 180.0
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align a single trial's estimates to truth by minimizing sum of squared wrapped errors.
    Tries Hungarian; falls back to brute-force if scipy isn't available.
    Returns (perm_truth_to_est, err_deg_aligned, doa_hat_aligned_deg).
    """
    try:
        return _align_hungarian(np.asarray(doa_hat_deg, float),
                                np.asarray(doa_true_deg, float),
                                period=period)
    except Exception:
        # No scipy or Hungarian failed → brute-force
        return _align_bruteforce(np.asarray(doa_hat_deg, float),
                                 np.asarray(doa_true_deg, float),
                                 period=period)


# ----------------- Summaries across trials (per-mode) -----------------
def bias_mse_var_with_perm(doa_hat_deg_trials: np.ndarray,
                           doa_true_deg: np.ndarray,
                           period: float = 180.0, sigma_hat_deg2_trials=None):
    """
    Compute per-mode signed bias (deg), MSE (deg²), and unbiased variance (deg²)
    using permutation alignment per trial and π-periodic wrapping.

    Parameters
    ----------
    doa_hat_deg_trials : (T, K) array  estimates in degrees, one row per trial
    doa_true_deg       : (K,)   array  ground-truth DOAs in degrees

    Returns
    -------
    stats : dict with fields
        'bias_deg'          : (K,) signed wrapped bias in deg
        'mse_deg2'          : (K,) per-mode MSE in deg² (mean of e^2)
        'var_unbiased_deg2' : (K,) per-mode unbiased variance in deg²
        'perm_history'      : (T, K) ints, chosen est index per truth mode (optional debugging)
    """
    """
     Aligns DOA estimates to truth per trial, wraps errors, and (optionally)
     permutes predicted per-DOA variances with the SAME permutation.

     Returns per-DOA bias, MSE, unbiased variance, permutation history, and
     (if provided) aligned σ̂² plus aggregated predicted std devs.
     """
    doa_hat_deg_trials = np.asarray(doa_hat_deg_trials, float)
    doa_true_deg = np.asarray(doa_true_deg, float)
    T, K = doa_hat_deg_trials.shape

    errs = np.zeros((T, K), dtype=float)  # wrapped, signed, in truth order
    perm_hist = np.zeros((T, K), dtype=int)

    aligned_sigma_hat_deg2 = None
    if sigma_hat_deg2_trials is not None:
        sigma_hat_deg2_trials = np.asarray(sigma_hat_deg2_trials, float)
        assert sigma_hat_deg2_trials.shape == (T, K)
        aligned_sigma_hat_deg2 = np.zeros((T, K), dtype=float)

    for t in range(T):
        # perm_t permutes θ̂ indices → truth order
        perm_t, err_t, _ = align_one_trial(doa_hat_deg_trials[t], doa_true_deg, period=period)
        perm_hist[t] = perm_t
        errs[t] = err_t

        # *** critical: carry the same permutation to σ̂² (θ̂-scale) ***
        if aligned_sigma_hat_deg2 is not None:
            aligned_sigma_hat_deg2[t] = sigma_hat_deg2_trials[t][perm_t]

    # Per-DOA aggregates over trials
    bias_deg = errs.mean(axis=0)  # E[e]
    mse_deg2 = (errs ** 2).mean(axis=0)  # E[e^2]
    if T > 1:
        var_unbiased_deg2 = (T / (T - 1.0)) * (mse_deg2 - bias_deg ** 2)
    else:
        var_unbiased_deg2 = np.zeros(K, dtype=float)
    var_unbiased_deg2 = np.maximum(var_unbiased_deg2, 0.0)

    out = dict(
        bias_deg=bias_deg,  # shape (K,)
        mse_deg2=mse_deg2,  # shape (K,)
        var_unbiased_deg2=var_unbiased_deg2,  # shape (K,)
        perm_history=perm_hist,  # shape (T, K)
        err_trials_deg=errs  # shape (T, K) (optional, handy)
    )

    if aligned_sigma_hat_deg2 is not None:
        # Predicted std per DOA = sqrt( E[σ̂²] )  (truth order)
        pred_std_hat_deg_per_doa = np.sqrt(aligned_sigma_hat_deg2.mean(axis=0))
        out.update(
            pred_var_hat_deg2_trials=aligned_sigma_hat_deg2,  # shape (T, K), truth order
            pred_std_hat_deg_per_doa=pred_std_hat_deg_per_doa,  # shape (K,)
            pred_std_hat_deg_mean=float(pred_std_hat_deg_per_doa.mean()),
            pred_std_hat_deg_median=float(np.sqrt(np.median(aligned_sigma_hat_deg2, axis=0)).mean())
        )

    return out

# ===================== Monte‑Carlo driver (wrapped, no permutation) =====================

def monte_carlo_compare_eq58_wrapped_no_perm(
    doas_deg, cohernet_signals=False, M=8, N=6000, fc=3.5e9, snr_db=-10.0,
    trials=30, shrinkage_rho=0.0, denom_eps=1e-6, seed=1
):
    """
    Monte‑Carlo without any permutation/assignment:
      mode i is always compared to true DOA i.

    Pipeline:
      • simulate Z
      • Rhat (no FB), optional small shrinkage
      • ESPRIT overlapped (Δ=λ/2), use λ "as is" (no |λ| projection)
      • θ̂ from principal phase, per mode (no unwrapping)
      • Eq.63 (Gaussian plug‑in) → Eq.66 → Eq.52/53 → Eq.58
      • empirical errors are 180°‑wrapped, compared by index (i→i)
      • empirical covariance via PyTorch (wrapped)

    Returns dict with empirical bias/variance/MSE (deg, deg²), the full (K×K) empirical
    covariance (deg²), predicted variances (deg²), and ratios (empirical variance / predicted).
    """
    rng = np.random.default_rng(seed)
    dsrc = len(doas_deg)
    true_deg = np.array(doas_deg, dtype=float)          # shape (K,)

    # Collect across trials
    theta_hat_all = []          # list of (K,) in deg
    pred_hat_deg2_all = []      # list of (K,) predicted var using θ̂ in scale
    pred_true_deg2_all = []     # list of (K,) predicted var using θ_true in scale

    for t in range(trials):
        # 1) simulate snapshots
        if cohernet_signals is True:
            Z = generate_signals_multi_cohernt(true_deg, M, N, fc, snr_db)
        else:
            Z = generate_signals_multi(true_deg, M, N, fc, snr_db)

        # 2) covariance (NO FB), optional shrinkage
        Rhat = shrinkage_cov(estimate_covariance_matrix(Z), rho=shrinkage_rho)

        # 3) ESPRIT (overlapped, shift=1), use λ "as is"
        esp = esprit_overlapped(Rhat, d_sources=dsrc, shift=1)
        J1, J2, E_x = esp["J1"], esp["J2"], esp["E_x"]
        lam, V, Q = esp["lambda"], esp["V"], esp["Q"]

        # DOA from principal phase only (no |λ| projection, no phase unwrapping)
        phi = np.angle(lam)                                            # (-π, π]
        theta_hat = np.arccos(np.clip(phi/np.pi, -1.0, 1.0))           # (rad)
        theta_hat_deg = np.rad2deg(theta_hat)                           # (deg), shape (K,)

        # 4) 4th‑order tensor (Gaussian plug‑in) and Eq.66 blocks
        Rcov_conj = rcov_conj_gaussian_plugin(Rhat, Ns=Z.shape[1])
        alpha_hat, S_hat = np.linalg.eigh(Rhat)
        sig_idx = np.argsort(alpha_hat)[-dsrc:]
        covHs_gh, covTs_gh = compute_delta_s_covariance_blocks_eq66(
            S_hat, alpha_hat, Rcov_conj, sig_idx, denom_eps=denom_eps
        )

        # 5) per‑mode Eq.52/53/58 (index‑aligned: i->i)
        pred_hat_deg2 = np.zeros(dsrc)
        pred_true_deg2 = np.zeros(dsrc)
        for i in range(dsrc):
            v_i = V[:, i][:, None]
            q_i = Q[i, :][None, :]
            lam_i = lam[i]

            eq52_i = compute_eq52_weighted(lam_i, q_i, E_x, J1, J2, covHs_gh, v_i)
            eq53_i = compute_eq53_weighted(lam_i, q_i, E_x, J1, J2, covTs_gh, v_i)

            var_hat  = compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_i, eq53_i, np.deg2rad(theta_hat_deg[i]), clip_nonneg=True)
            var_true = compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_i, eq53_i, np.deg2rad(true_deg[i]),        clip_nonneg=True)

            pred_hat_deg2[i]  = var_hat  * (180/np.pi)**2
            pred_true_deg2[i] = var_true * (180/np.pi)**2

        theta_hat_all.append(theta_hat_deg)
        pred_hat_deg2_all.append(pred_hat_deg2)
        pred_true_deg2_all.append(pred_true_deg2)

    # Stack and summarize
    TH = np.vstack(theta_hat_all)                                  # (T, K) in deg
    TT = np.tile(true_deg, (TH.shape[0], 1))                       # (T, K) in deg
    """
    # 180°‑wrapped errors per trial, per mode
    Ewrap = wrap_diff_deg(TH, TT, period=180.0)                    # (T, K)
    """
    mspe = [MSPE_Empricial(TH[index], TT[index]) for index in range(TH.shape[0]) ]

    bias = [BPE(TH[index], TT[index]) for index in range(TH.shape[0])]
    """
    # --- Bias / Variance / MSE per mode (wrapped)
    T = Ewrap.shape[0]
    bias_deg = Ewrap.mean(axis=0)                                  # (K,)
    var_deg2_unbiased = Ewrap.var(axis=0, ddof=1)                  # (K,)
    var_deg2_biased   = Ewrap.var(axis=0, ddof=0)                  # (K,)
    mse_deg2 = (Ewrap**2).mean(axis=0)                             # (K,)
    # Identity checks:
    mse_from_parts = var_deg2_biased + bias_deg**2
    mse_from_parts_unbiased = ((T-1)/T)*var_deg2_unbiased + bias_deg**2

    # Empirical covariance (deg²) with wrapping, via torch
    emp_cov_deg2 = empirical_error_cov(
        torch.tensor(TH, dtype=torch.float64),
        torch.tensor(TT, dtype=torch.float64),
        period=180.0, unbiased=True, return_scalar_if_K1=False
    ).detach().cpu().numpy()
    """
    stats = bias_mse_var_with_perm(TH, true_deg, period=180.0)

    pred_hat_m = np.mean(np.vstack(pred_hat_deg2_all), axis=0)     # (K,)
    pred_true_m = np.mean(np.vstack(pred_true_deg2_all), axis=0)   # (K,)

    variance = [mspe[i] - (bias[i] ** 2.0) for i in range(len(mspe))]

    summary = {
        "true_DOAs_deg": list(map(float, true_deg)),
        #"empirical_bias_deg_wrapped_no_perm": bias_deg.tolist(),
        #"empirical_var_deg2_wrapped_no_perm": var_deg2_unbiased.tolist(),
        #"empirical_mse_deg2_wrapped_no_perm": mse_deg2.tolist(),
        #"empirical_cov_deg2_wrapped_no_perm": emp_cov_deg2.tolist(),
        "predicted_var_deg2_scale_theta_hat": pred_hat_m.tolist(),
        "predicted_var_deg2_scale_theta_true": pred_true_m.tolist(),
        # Compare **variance** (not MSE) to Eq.(58)
        #"ratio_empVar_over_pred_hat": (var_deg2_unbiased / pred_hat_m).tolist(),
        #"ratio_empVar_over_pred_true": (var_deg2_unbiased / pred_true_m).tolist(),
        # Consistency checks for MSE decomposition
        #"mse_vs_var_bias_check": {
        #    "mse": mse_deg2.tolist(),
        #    "var_biased_plus_bias2": mse_from_parts.tolist(),
        #    "((T-1)/T)*var_unbiased_plus_bias2": mse_from_parts_unbiased.tolist(),
        #    "T": int(T),
        #},
        "bias1" : stats["bias_deg"],
        "mspe1" : stats["mse_deg2"],
        "emprical_var1" : stats["var_unbiased_deg2"],
        "trials": int(trials), "M": int(M), "N": int(N), "SNR_dB": float(snr_db),
        "mspe" : mspe,
        "mspe_mean" : np.mean(mspe),
        "bias" : bias,
        "bias_mean" : np.mean(bias),
        "emprical_var" : variance,
        "emprical_var_mean": np.mean(variance)

    }
    return summary



def model_based_dnn_uncertainty(
    doas_deg, Rx, signal_shape, true_doa
):
    """
    Monte‑Carlo without any permutation/assignment:
      mode i is always compared to true DOA i.

    Pipeline:
      • simulate Z
      • Rhat (no FB), optional small shrinkage
      • ESPRIT overlapped (Δ=λ/2), use λ "as is" (no |λ| projection)
      • θ̂ from principal phase, per mode (no unwrapping)
      • Eq.63 (Gaussian plug‑in) → Eq.66 → Eq.52/53 → Eq.58
      • empirical errors are 180°‑wrapped, compared by index (i→i)
      • empirical covariance via PyTorch (wrapped)

    Returns dict with empirical bias/variance/MSE (deg, deg²), the full (K×K) empirical
    covariance (deg²), predicted variances (deg²), and ratios (empirical variance / predicted).
    """

    # 2) covariance (NO FB), optional shrinkage
    Rhat = Rx.detach().cpu().numpy()
    doas_deg = doas_deg.detach().cpu().numpy()
    true_doa = true_doa.detach().cpu().numpy()

    dsrc = len(doas_deg)
    true_deg = np.array(true_doa, dtype=float)          # shape (K,)

    # Collect across trials
    theta_hat_all = []          # list of (K,) in deg
    pred_hat_deg2_all = []      # list of (K,) predicted var using θ̂ in scale
    pred_true_deg2_all = []     # list of (K,) predicted var using θ_true in scale




    # 3) ESPRIT (overlapped, shift=1), use λ "as is"
    esp = esprit_overlapped(Rhat, d_sources=dsrc, shift=1)
    J1, J2, E_x = esp["J1"], esp["J2"], esp["E_x"]
    lam, V, Q = esp["lambda"], esp["V"], esp["Q"]

    # DOA from principal phase only (no |λ| projection, no phase unwrapping)
    theta_hat_deg = doas_deg

    # 4) 4th‑order tensor (Gaussian plug‑in) and Eq.66 blocks
    Rcov_conj = rcov_conj_gaussian_plugin(Rhat, Ns=signal_shape)
    alpha_hat, S_hat = np.linalg.eigh(Rhat)
    sig_idx = np.argsort(alpha_hat)[-dsrc:]
    covHs_gh, covTs_gh = compute_delta_s_covariance_blocks_eq66(
        S_hat, alpha_hat, Rcov_conj, sig_idx, denom_eps=1e-6
    )

    # 5) per‑mode Eq.52/53/58 (index‑aligned: i->i)
    pred_hat_deg2 = np.zeros(dsrc)
    pred_true_deg2 = np.zeros(dsrc)
    for i in range(dsrc):
        v_i = V[:, i][:, None]
        q_i = Q[i, :][None, :]
        lam_i = lam[i]

        eq52_i = compute_eq52_weighted(lam_i, q_i, E_x, J1, J2, covHs_gh, v_i)
        eq53_i = compute_eq53_weighted(lam_i, q_i, E_x, J1, J2, covTs_gh, v_i)

        var_hat  = compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_i, eq53_i, np.deg2rad(theta_hat_deg[i]), clip_nonneg=True)
        var_true = compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_i, eq53_i, np.deg2rad(true_deg[i]),        clip_nonneg=True)

        pred_hat_deg2[i]  = var_hat  * (180/np.pi)**2
        pred_true_deg2[i] = var_true * (180/np.pi)**2

    theta_hat_all.append(theta_hat_deg)
    pred_hat_deg2_all.append(pred_hat_deg2)
    pred_true_deg2_all.append(pred_true_deg2)

    # Stack and summarize
    TH = np.vstack(theta_hat_all)                                  # (T, K) in deg
    TT = np.tile(true_deg, (TH.shape[0], 1))                       # (T, K) in deg
    """
    # 180°‑wrapped errors per trial, per mode
    Ewrap = wrap_diff_deg(TH, TT, period=180.0)                    # (T, K)
    """
    mspe = [MSPE_Empricial(TH[index], TT[index]) for index in range(TH.shape[0]) ]

    bias = [BPE(TH[index], TT[index]) for index in range(TH.shape[0])]

    stats = bias_mse_var_with_perm(TH, true_deg, period=180.0)

    pred_hat_m = np.mean(np.vstack(pred_hat_deg2_all), axis=0)     # (K,)
    pred_true_m = np.mean(np.vstack(pred_true_deg2_all), axis=0)   # (K,)

    variance = [mspe[i] - (bias[i] ** 2.0) for i in range(len(mspe))]

    summary = {
        "true_DOAs_deg": list(map(float, true_deg)),
        #"empirical_bias_deg_wrapped_no_perm": bias_deg.tolist(),
        #"empirical_var_deg2_wrapped_no_perm": var_deg2_unbiased.tolist(),
        #"empirical_mse_deg2_wrapped_no_perm": mse_deg2.tolist(),
        #"empirical_cov_deg2_wrapped_no_perm": emp_cov_deg2.tolist(),
        "predicted_var_deg2_scale_theta_hat": pred_hat_m.tolist(),
        "predicted_var_deg2_scale_theta_true": pred_true_m.tolist(),
        # Compare **variance** (not MSE) to Eq.(58)
        #"ratio_empVar_over_pred_hat": (var_deg2_unbiased / pred_hat_m).tolist(),
        #"ratio_empVar_over_pred_true": (var_deg2_unbiased / pred_true_m).tolist(),
        # Consistency checks for MSE decomposition
        #"mse_vs_var_bias_check": {
        #    "mse": mse_deg2.tolist(),
        #    "var_biased_plus_bias2": mse_from_parts.tolist(),
        #    "((T-1)/T)*var_unbiased_plus_bias2": mse_from_parts_unbiased.tolist(),
        #    "T": int(T),
        #},
        "bias1" : stats["bias_deg"],
        "mspe1" : stats["mse_deg2"],
        "emprical_var1" : stats["var_unbiased_deg2"],
        #"trials": int(trials), "M": int(M), "N": int(N), "SNR_dB": float(snr_db),
        "mspe" : mspe,
        "mspe_mean" : np.mean(mspe),
        "bias" : bias,
        "bias_mean" : np.mean(bias),
        "emprical_var" : variance,
        "emprical_var_mean": np.mean(variance)

    }
    return summary





class UncertaintyEstimation(nn.Module):
    def __init__(self, signal_shape):
        super().__init__()
        self.__signal_shape = signal_shape

    def compute_predicated_uncertainty(self, doas_deg, Rx):
        """
        Given the math formulation calculate the uncertainty

        Pipeline:
          • simulate Z
          • Rhat (no FB), optional small shrinkage
          • ESPRIT overlapped (Δ=λ/2), use λ "as is" (no |λ| projection)
          • θ̂ from principal phase, per mode (no unwrapping)
          • Eq.63 (Gaussian plug‑in) → Eq.66 → Eq.52/53 → Eq.58
          • empirical errors are 180°‑wrapped, compared by index (i→i)
          • empirical covariance via PyTorch (wrapped)

        Returns dict with empirical bias/variance/MSE (deg, deg²), the full (K×K) empirical
        covariance (deg²), predicted variances (deg²), and ratios (empirical variance / predicted).
        """

        # 2) covariance (NO FB), optional shrinkage
        Rhat = Rx.detach().cpu().numpy()
        doas_deg = doas_deg.detach().cpu().numpy()

        dsrc = len(doas_deg)

        # Collect across trials
        theta_hat_all = []  # list of (K,) in deg
        pred_hat_deg2_all = []  # list of (K,) predicted var using θ̂ in scale

        # 3) ESPRIT (overlapped, shift=1), use λ "as is"
        esp = esprit_overlapped(Rhat, d_sources=dsrc, shift=1)
        J1, J2, E_x = esp["J1"], esp["J2"], esp["E_x"]
        lam, V, Q = esp["lambda"], esp["V"], esp["Q"]

        # DOA from principal phase only (no |λ| projection, no phase unwrapping)
        theta_hat_deg = doas_deg

        # 4) 4th‑order tensor (Gaussian plug‑in) and Eq.66 blocks
        Rcov_conj = rcov_conj_gaussian_plugin(Rhat, Ns=self.__signal_shape)
        alpha_hat, S_hat = np.linalg.eigh(Rhat)
        sig_idx = np.argsort(alpha_hat)[-dsrc:]
        covHs_gh, covTs_gh = compute_delta_s_covariance_blocks_eq66(
            S_hat, alpha_hat, Rcov_conj, sig_idx, denom_eps=1e-6
        )

        # 5) per‑mode Eq.52/53/58 (index‑aligned: i->i)
        pred_hat_deg2 = np.zeros(dsrc)
        for i in range(dsrc):
            v_i = V[:, i][:, None]
            q_i = Q[i, :][None, :]
            lam_i = lam[i]

            eq52_i = compute_eq52_weighted(lam_i, q_i, E_x, J1, J2, covHs_gh, v_i)
            eq53_i = compute_eq53_weighted(lam_i, q_i, E_x, J1, J2, covTs_gh, v_i)

            var_hat = compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_i, eq53_i, np.deg2rad(theta_hat_deg[i]),
                                                              clip_nonneg=True)

            pred_hat_deg2[i] = var_hat * (180 / np.pi) ** 2

        theta_hat_all.append(theta_hat_deg)
        pred_hat_deg2_all.append(pred_hat_deg2)

        pred_hat_m = np.mean(np.vstack(pred_hat_deg2_all), axis=0)  # (K,)

        return torch.Tensor(pred_hat_m)

    def forward(self, doas_deg, Rx):
        """
        Return the Std Deviation of the uncertainty
        :param doas_deg: The predicated doa's used as part of uncertainty calculation (as close as to boresight the better accuracy)
        :param Rx: The calculated coveriance matrix of the signal
        :return: StdDeviation of the uncertainty
        """
        # Run in batch mode
        uncertainty = torch.zeros_like(doas_deg)
        for index in range(doas_deg.shape[0]):
            uncertainty[index] = self.compute_predicated_uncertainty(doas_deg[index], Rx[index])
        return np.sqrt(uncertainty)
