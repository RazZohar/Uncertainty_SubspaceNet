"""Subspace-Net main script
    Details
    -------
    Name: main.py
    Authors: R Zohar
    Created: 01/10/21
    Edited: 01/12/24

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets
        * training: For training DR-MUSIC model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator

from src.methods import MUSIC, RootMUSIC, Esprit, MVDR

import src.create_codebook as codebook_creation

import src.qunatizer as quantizer

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
                           period: float = 180.0):
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
    doa_hat_deg_trials = np.asarray(doa_hat_deg_trials, float)
    doa_true_deg = np.asarray(doa_true_deg, float)
    T, K = doa_hat_deg_trials.shape

    # Collect aligned, wrapped errors per trial in truth order
    errs = np.zeros((T, K), dtype=float)
    perm_hist = np.zeros((T, K), dtype=int)

    for t in range(T):
        perm_t, err_t, _ = align_one_trial(doa_hat_deg_trials[t], doa_true_deg, period=period)
        perm_hist[t] = perm_t
        errs[t] = err_t  # wrapped, signed, truth order

    # Per-mode stats
    bias_deg = errs.mean(axis=0)                     # signed bias (deg)
    mse_deg2 = errs**2              # MSE (deg²)
    # Unbiased sample variance around the (signed) mean:
    # s^2 = (T/(T-1)) * (MSE - bias^2)
    var_unbiased_deg2 =  (mse_deg2 - bias_deg**2)
    # Clamp tiny negatives from rounding
    var_unbiased_deg2 = np.maximum(var_unbiased_deg2, 0.0)

    return dict(
        bias_deg=bias_deg,
        mse_deg2=mse_deg2,
        var_unbiased_deg2=var_unbiased_deg2,
        perm_history=perm_hist
    )

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


def model_based_uncertainty(
    doas_deg, signals, predicated_doa,
    trials=1, shrinkage_rho=0.0, denom_eps=1e-6, seed=1
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
    signals = signals.detach().cpu().numpy()
    doas_deg = doas_deg.detach().cpu().numpy()
    predicated_doa = predicated_doa.detach().cpu().numpy()


    dsrc = len(doas_deg)
    true_deg = np.array(doas_deg, dtype=float)          # shape (K,)

    # Collect across trials
    theta_hat_all = []          # list of (K,) in deg
    pred_hat_deg2_all = []      # list of (K,) predicted var using θ̂ in scale
    pred_true_deg2_all = []     # list of (K,) predicted var using θ_true in scale


    # 1) Snapshots are given via signals
    # 2) covariance (NO FB), optional shrinkage
    Rhat = shrinkage_cov(estimate_covariance_matrix(signals), rho=shrinkage_rho)

    # 3) ESPRIT (overlapped, shift=1), use λ "as is"
    esp = esprit_overlapped(Rhat, d_sources=dsrc, shift=1)
    J1, J2, E_x = esp["J1"], esp["J2"], esp["E_x"]
    lam, V, Q = esp["lambda"], esp["V"], esp["Q"]

    # DOA from principal phase only (no |λ| projection, no phase unwrapping)
    #phi = np.angle(lam)
    #phi = (phi + np.pi/2) % np.pi - np.pi/2# (-π, π]
    #theta_hat = np.arccos(np.clip(phi/np.pi, -1.0, 1.0))           # (rad)
    #theta_hat_deg = np.rad2deg(theta_hat)                           # (deg), shape (K,)
    theta_hat_deg = predicated_doa
    # 4) 4th‑order tensor (Gaussian plug‑in) and Eq.66 blocks
    Rcov_conj = rcov_conj_gaussian_plugin(Rhat, Ns=signals.shape[1])
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
        #"trials": int(trials), "M": int(M), "N": int(N), "SNR_dB": float(snr_db),
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


from itertools import cycle
from matplotlib import rcParams
from cycler import cycler

def run_simulation_by_system_params(system_model_params):
    # Generate model configuration
    MAXIMAL_TAU = 8
    model_config = (
        ModelGenerator()
        .set_model_type("SignalsSubspaceNet")  # "TaskIgnorantSubspaceNet", SignalsSubspaceNet
        .set_diff_method("esprit")
        .set_tau(min(MAXIMAL_TAU, system_model_params.T - 1))
        .set_model(system_model_params)
    )
    print('Generating model configuration')
    # Define samples size
    samples_size = 100000  # Overall dateset size
    train_test_ratio = 0.05  # training and testing datasets ratio
    model_config.tau = 8
    # Sets simulation filename
    simulation_filename = get_simulation_filename(
        system_model_params=system_model_params, model_config=model_config
    )
    print(f'Set model configuration and export to configuration/{simulation_filename}')
    system_model_json = system_model_params.export_to_json()
    with open(f'configuration/{simulation_filename}.json', "w") as outfile:
        outfile.write(system_model_json)
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    # Initialize seed
    set_unified_seed()
    # Datasets creation
    if commands["CREATE_DATA"]:
        # Define which datasets to generate
        create_training_data = True  # Flag for creating training data
        create_testing_data = True  # Flag for creating test data
        print("Creating Data...")
        if create_training_data:
            # Generate training dataset
            # if model_config.model_type == "SignalsSubspaceNet" or model_config.model_type == "TaskIgnorantSubspaceNet":
            #    model_type_name = "SubspaceNet"
            # else:
            #    model_type_name = model_config.model_type
            model_type_name = model_config.model_type
            train_dataset, generic_train_dataset, samples_model = create_dataset(
                system_model_params=system_model_params,
                samples_size=samples_size,
                model_type=model_type_name,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="train",
            )
        if create_testing_data:
            # Generate test dataset
            test_dataset, generic_test_dataset, samples_model = create_dataset(
                system_model_params=system_model_params,
                samples_size=int(train_test_ratio * samples_size),
                model_type=model_config.model_type,
                tau=model_config.tau,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=None,
                phase="test",
            )
    # Datasets loading
    elif commands["LOAD_DATA"]:
        # if model_config.model_type == "SignalsSubspaceNet" or model_config.model_type == "TaskIgnorantSubspaceNet":
        #    model_type_name = "SubspaceNet"
        # else:
        #    model_type_name = model_config.model_type
        model_type_name = model_config.model_type

        (
            train_dataset,
            test_dataset,
            generic_test_dataset,
            samples_model,
            generic_train_dataset,
        ) = load_datasets(
            system_model_params=system_model_params,
            model_type=model_type_name,
            samples_size=samples_size,
            datasets_path=datasets_path,
            train_test_ratio=train_test_ratio,
            is_training=True,
        )
    # Training stage special model
    if commands["TRAIN_MODEL_SOURCES"]:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_batch_size(1024)
            .set_epochs(40)
            .set_model(model=model_config)
            .set_optimizer(optimizer="Adam", learning_rate=0.001, weight_decay=1e-5)
            .set_training_dataset(generic_train_dataset)
            .set_schedular(step_size=20, gamma=0.2)
            .set_criterion()
        )

        # Update to handle root music with cohernt sources
        # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

        if commands["LOAD_MODEL"]:
            simulation_parameters.load_model(
                loading_path=saving_path / "final_models" / simulation_filename
            )
        # Print training simulation details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            parameters=simulation_parameters,
            phase="training",
        )

        # Perform simulation training and evaluation stages
        #        with profile(activities=[ProfilerActivity.CPU]) as prof:
        #            with record_function("model_inference"):
        model, loss_train_list, loss_valid_list = train(
            training_parameters=simulation_parameters,
            model_name=simulation_filename,
            saving_path=saving_path,
        )

        # print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

        # Save model weights
        if commands["SAVE_MODEL"]:
            simulation_filename = simulation_filename
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(simulation_filename),
            )
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # For this purpose we use evaluate
        # evaluate_model_command()
    # Evaluation stage
    if commands["EVALUATE_MODE_SOURCES"]:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions("rmse")
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
            )
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        generic_test_dataset = torch.utils.data.DataLoader(
            generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        # Load pre-trained model
        if not commands["TRAIN_MODEL_SOURCES"]:
            # simulation_filename = simulation_filename + f'_VQVAE_Quantized_{CODEBOOK_SIZE}' + '_Quantized_Trained_{codebook_size}'.format(
            #    codebook_size=CODEBOOK_SIZE)

            # Define an evaluation parameters instance
            simulation_parameters = (
                TrainingParams()
                .set_model(model=model_config)
                .load_model(
                    loading_path=saving_path
                                 / "final_models"
                                 / simulation_filename
                )
            )
            model = simulation_parameters.model
        # print simulation summary details
        simulation_summary(
            system_model_params=system_model_params,
            model_type=model_config.model_type,
            phase="evaluation",
            parameters=simulation_parameters,
        )
        # Evaluate DNN models, augmented and subspace methods
        evaluate(
            model=model,
            model_type=model_config.model_type,
            model_test_dataset=generic_test_dataset,
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            subspace_criterion=subspace_criterion,
            system_model=samples_model,
            figures=figures,
            plot_spec=plot_spectrum_flag,
        )
    # Check Uncertainty
    if commands["EVALUATE_UNCERTAINTY_MODEL"]:
        # Initialize figures dict for plotting
        figures = initialize_figures()
        # Define loss measure for evaluation
        criterion, subspace_criterion = set_criterions("rmse")
        # Load datasets for evaluation
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            test_dataset, generic_test_dataset, samples_model = load_datasets(
                system_model_params=system_model_params,
                model_type=model_config.model_type,
                samples_size=samples_size,
                datasets_path=datasets_path,
                train_test_ratio=train_test_ratio,
            )
        # Generate DataLoader objects
        model_test_dataset = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, drop_last=False
        )
        generic_test_dataset = torch.utils.data.DataLoader(
            generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
        )

        simulation_parameters = (
            TrainingParams()
            .set_model(model=model_config)
            .load_model(
                loading_path=saving_path
                             / "final_models"
                             / simulation_filename
            )
        )
        model = simulation_parameters.model
        test_length = 0
        result_uncertainty_all = []
        results_uncertainty_mb_all = []
        esprit_model_based = Esprit(SystemModel(system_model_params=system_model_params))
        with torch.no_grad():
            for data in generic_test_dataset:
                X, DOA = data
                test_length += DOA.shape[0]
                # Convert observations and DoA to device
                X = X.to(device)
                DOA = DOA.to(device)
                # Get model output
                model_output = model(X)
                predicates_doa, R, _ = model_output

                results_uncertainty = model_based_dnn_uncertainty(np.rad2deg(predicates_doa)[0], R.squeeze(dim=0),
                                                                  X.shape[-1], np.rad2deg(DOA)[0])
                result_uncertainty_all.append(results_uncertainty)

                doa_predicated_model_based, _ = esprit_model_based.narrowband(X.squeeze(dim=0), system_model_params.M)
                results_uncertainty_model_based = model_based_uncertainty(np.rad2deg(DOA)[0], X.squeeze(dim=0), np.rad2deg(doa_predicated_model_based))
                results_uncertainty_mb_all.append(results_uncertainty_model_based)

        result_uncertainty_avg = {
            k: np.mean([d[k] for d in result_uncertainty_all], axis=0)
            for k in result_uncertainty_all[0]
        }
        result_uncertainty_mb_avg = {
            k: np.mean([d[k] for d in results_uncertainty_mb_all], axis=0)
            for k in results_uncertainty_mb_all[0]
        }

        result_uncertainty_avg['emprical_var1'] = result_uncertainty_avg['mspe1'] - result_uncertainty_avg['bias1'] ** 2
        result_uncertainty_mb_avg['emprical_var1'] = result_uncertainty_mb_avg['mspe1'] - result_uncertainty_mb_avg['bias1'] ** 2

        print(result_uncertainty_avg)
        print(result_uncertainty_mb_avg)
        print(
            f'Std Dev - Empricial {np.sqrt(result_uncertainty_avg['emprical_var1'])} vs StdDev Predicated {np.sqrt(result_uncertainty_avg["predicted_var_deg2_scale_theta_hat"])}')

        print(
            f'Model Based Std Dev - Empricial {np.sqrt(result_uncertainty_mb_avg['emprical_var1'])} vs StdDev Predicated {np.sqrt(result_uncertainty_mb_avg["predicted_var_deg2_scale_theta_hat"])}')

        uncertainty_result_file_name = simulation_filename + '_uncertainty.npy'
        np.save(saving_path / uncertainty_result_file_name, {"MBDL" : result_uncertainty_avg, "MB":result_uncertainty_mb_avg})

    plt.show()
    print("end")

    return {"MBDL" : result_uncertainty_avg, "MB":result_uncertainty_mb_avg}

def sweep_simulation_by_snr(snr, M=2, coherent_case=False):
    #global model_config
    if coherent_case is True:
        signal_nature = "coherent"
    else:
        signal_nature = "non-coherent"
    # Define system model parameters
    system_model_params = (
        SystemModelParams()
        .set_parameter("N", 8)
        .set_parameter("M", M)
        .set_parameter("T", 500)
        .set_parameter("snr", snr)
        .set_parameter("signal_type", "NarrowBand")
        .set_parameter("signal_nature", signal_nature)
        .set_parameter("eta", 0)
        .set_parameter("bias", 0.0)
        .set_parameter("sv_noise_var", 0)
        .set_parameter("codebook_size", CODEBOOK_SIZE)
    )

    return run_simulation_by_system_params(system_model_params)

def sweep_simulation_by_eta(eta_list):
    now = datetime.now()
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")

    #global model_config
    results = {}
    for eta in eta_list:
        # Define system model parameters
        system_model_params = (
            SystemModelParams()
            .set_parameter("N", 8)
            .set_parameter("M", 2)
            .set_parameter("T", 100)
            .set_parameter("snr", 3)
            .set_parameter("signal_type", "NarrowBand")
            .set_parameter("signal_nature", "non-coherent")
            .set_parameter("eta", eta)
            .set_parameter("bias", 0.0)
            .set_parameter("sv_noise_var", 0)
            .set_parameter("codebook_size", CODEBOOK_SIZE)
        )

        results[eta] = run_simulation_by_system_params(system_model_params)

    results_file_name = dt_string_for_save + "results_file_uncertainty_MBDL_eta_sweep.npy"
    np.save(saving_path / results_file_name, results)

    create_figures_from_data(results, title=f"Uncertainty Predicated & Emprical - Coherent case M={2}", argument="Eta")

 # --- collect mean values ---
def collect_method_means(data_dict, method_key):
    snrs_sorted = sorted(data_dict.keys())
    pred = []
    emp = []
    for snr in snrs_sorted:
        block = data_dict[snr][method_key]
        pred.append(np.sqrt(block['predicted_var_deg2_scale_theta_hat']).mean())
        emp.append(np.sqrt(block['emprical_var1']).mean())
    return snrs_sorted, np.array(pred), np.array(emp)

def create_figures_from_data(dict_data, title, argument="SNR (dB)"):
    snrs = sorted(dict_data.keys())



    # Collect for both methods
    snrs, mbdl_pred, mbdl_emp = collect_method_means(dict_data, "MBDL")
    _, mb_pred, mb_emp = collect_method_means(dict_data, "MB")

    # --- plot in your preferred style ---
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(snrs, mbdl_pred, linestyle='--', marker='o', label="MBDL Predicted")
    ax.plot(snrs, mbdl_emp, linestyle='--', marker='*', label="MBDL Empirical")
    ax.plot(snrs, mb_pred, linestyle='--', marker='o', label="MB Predicted")
    ax.plot(snrs, mb_emp, linestyle='--', marker='*', label="MB Empirical")

    # Title + legend
    #ax.set_title("Predicted vs Empirical DOA stddev vs " + argument, pad=30)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5),
              ncol=2, frameon=False)

    ax.set_xlabel(argument)
    ax.set_ylabel("Std-dev (deg)")
    ax.grid(True, which='both')
    plt.tight_layout()

    # --- inset zoom over SNR [0,20] ---
    snrs_arr = np.asarray(snrs)
    mask = (snrs_arr >= 0) & (snrs_arr <= 20)

    axins = inset_axes(ax, width="35%", height="35%", loc="upper right", borderpad=1.0)
    axins.plot(snrs, mbdl_pred, linestyle='--', marker='o')
    axins.plot(snrs, mbdl_emp, linestyle='--', marker='*')
    axins.plot(snrs, mb_pred, linestyle='--', marker='o')
    axins.plot(snrs, mb_emp, linestyle='--', marker='*')

    axins.set_xlim(0, 20)
    ys_inset = np.concatenate([mbdl_pred[mask], mbdl_emp[mask], mb_pred[mask], mb_emp[mask]])
    pad = 0.05 * (ys_inset.max() - ys_inset.min() + 1e-12)
    axins.set_ylim(0, 3 + pad)
    #axins.set_ylim(ys_inset.min() - pad, ys_inset.max() + pad)

    axins.tick_params(labelsize=6)
    ax.indicate_inset_zoom(axins, edgecolor="0.5")

    # Save
    plt.savefig("stddev_means.png", dpi=300, bbox_inches="tight")
    plt.savefig("stddev_means.pdf", dpi=300, bbox_inches="tight")
    plt.show()


    # Collect values (mean across both DOAs)
    """MBDL_empirical = [np.sqrt(dict_data[snr]['MBDL']['emprical_var1']).mean() for snr in snrs]
    MBDL_predicted = [np.sqrt(dict_data[snr]['MBDL']['predicted_var_deg2_scale_theta_hat']).mean() for snr in snrs]

    MB_empirical = [np.sqrt(dict_data[snr]['MB']['emprical_var1']).mean() for snr in snrs]
    MB_predicted = [np.sqrt(dict_data[snr]['MB']['predicted_var_deg2_scale_theta_hat']).mean() for snr in snrs]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, MBDL_empirical, marker='o', label="MBDL Empirical StdDev")
    plt.plot(snrs, MBDL_predicted, marker='o', label="MBDL Predicted StdDev")
    plt.plot(snrs, MB_empirical, marker='s', label="MB Empirical StdDev")
    plt.plot(snrs, MB_predicted, marker='s', label="MB Predicted StdDev")

    plt.xlabel("SNR (dB)")
    plt.ylabel("Std Dev (degrees)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save figure
    plt.savefig("stddev_comparison.png", dpi=300, bbox_inches="tight")

    plt.show()
    """


def sweep_uncertainty_snrs(snrs=[-10.0, -3.0, 0.0, 3.0, 10.0, 20.0], M=2, coherent_case=False):
    results = {}
    for snr in snrs:
        summary = sweep_simulation_by_snr(snr, M=M, coherent_case=coherent_case)
        results[snr] = summary

        print(f"\nSNR = {snr:>5.1f} dB")
        # print("  bias [deg] (per mode):", summary["empirical_bias_deg_wrapped_no_perm"])
        # print("  var  [deg^2] (per mode):", summary["empirical_var_deg2_wrapped_no_perm"])
        # print("Emprical bias is (θ̂ scale) : ", summary["bias"])
        # print("Emprical MSPE (θ̂ scale) : ", summary["mspe"])
        # print("Emprical Variance (θ̂ scale) : ", summary["emprical_var"])
        # print("Emprical bias is #1 (θ̂ scale) : ", summary["bias1"])
        # print("Emprical MSPE (θ̂ scale) #1  : ", summary["mspe1"])
        for value in summary.values():
            print("Emprical Variance (θ̂ scale) #1  : ", value["emprical_var1"])
            print("Emprical Variance Mean (θ̂ scale) : ", value["emprical_var_mean"])
            print("  pred [deg^2] (θ̂ scale):", value["predicted_var_deg2_scale_theta_hat"])
            # print("  ratio empVar/pred  :", summary["ratio_empVar_over_pred_hat"])
    results_file_name = dt_string_for_save + "results_file_uncertainty_MBDL_snr_sweep.npy"
    np.save(saving_path / results_file_name, results)


    create_figures_from_data(results, title=f"Uncertainty Predicated & Emprical - Coherent case M={2}")





if __name__ == "__main__":
    # Initialize paths
    external_data_path = Path.cwd() / "data"
    scenario_data_path = "uncertainty"
    datasets_path = external_data_path / "datasets" / scenario_data_path
    simulations_path = external_data_path / "simulations"
    saving_path = external_data_path / "weights"
    # create folders if not exists
    datasets_path.mkdir(parents=True, exist_ok=True)
    (datasets_path / "train").mkdir(parents=True, exist_ok=True)
    (datasets_path / "test").mkdir(parents=True, exist_ok=True)
    datasets_path.mkdir(parents=True, exist_ok=True)
    simulations_path.mkdir(parents=True, exist_ok=True)
    saving_path.mkdir(parents=True, exist_ok=True)
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # Operations commands
    commands = {
        "SAVE_TO_FILE": False,  # Saving results to file or present them over CMD
        "CREATE_DATA": True,  # Creating new dataset
        "LOAD_DATA": True,  # Loading data from exist dataset
        "LOAD_MODEL": False,  # Load specific model for training
        "TRAIN_MODEL": False,  # Applying training operation
        "SAVE_MODEL": True,  # Saving tuned model
        "EVALUATE_MODE": False,  # Evaluating desired algorithms
        "CREATE_CODEBOOK": False,  # Create the codebook for VQ-VAE
        "TRAIN_QUANTIZED": False,  # Train the model for the quantization
        "TRAIN_SCALAR_QUANTIZATION": False,  # Train the model for Scalar quantization

        # Source - task based quantization
        "TRAIN_MODEL_SOURCES": True,  # Applying training operation for the sources
        "EVALUATE_MODE_SOURCES": False,  # Evaluating desired algorithms
        "CREATE_CODEBOOK_SOURCES": False,  # Create the codebook for VQ-VAE
        "TRAIN_QUANTIZED_SOURCES": False,  # Train the model for the quantization

        "TRAIN_SCALAR_QUANTIZATION_SOURCES": False,  # Train the model for Scalar quantization

        "EVALUATE_UNCERTAINTY_MODEL": True
    }

    ## Graph tools
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dash_styles = [
        '-',
        (0, (5, 5)),
        (0, (1, 5)),
        (0, (3, 5, 1, 5)),
        (0, (5, 1, 1, 1))
    ]
    # Extend dash styles to match color length
    dash_cycle = list(itertools.islice(itertools.cycle(dash_styles), len(colors)))

    # Now same length, can combine
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors) + cycler(linestyle=dash_cycle)
    CODEBOOK_SIZE = 128

    print(f'Start Executing commands')
    # Saving simulation scores to external file
    if commands["SAVE_TO_FILE"]:
        file_path = (
                simulations_path / "results" / "scores" / Path(dt_string_for_save + ".txt")
        )
        sys.stdout = open(file_path, "w")

    #[-3.0, 0.0, 3.0, 10.0, 20.0]
    sweep_uncertainty_snrs(snrs=[-3.0, 0.0, 3.0, 10.0, 20.0], M=2, coherent_case=False)
    sweep_uncertainty_snrs(snrs=[-3.0, 0.0, 3.0, 10.0, 20.0], M=2, coherent_case=True)
    sweep_simulation_by_eta([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])

    sweep_uncertainty_snrs(snrs=[-3.0, 0.0, 3.0, 10.0, 20.0], M=3, coherent_case=False)
    sweep_uncertainty_snrs(snrs=[-3.0, 0.0, 3.0, 10.0, 20.0], M=3, coherent_case=True)

    #sweep_simulation_by_eta([0.2, 0.15, 0.1, 0.05, 0.01])

    # ============================== Demo run ==============================

def model_based_uncertainty():
    cohernet_signals = False
    np.random.seed(123)
    doas = [20.0, 40.0]      # two sources; mode 0 ↔ 10°, mode 1 ↔ 25°

    # Example: sweep SNRs and print summary lines
    TIME_WINDOW = 1000
    snrs = [-10.0, -3.0, 0.0, 3.0, 10.0, 20.0]
    results = {}
    for snr in snrs:
        summary = monte_carlo_compare_eq58_wrapped_no_perm(
            doas_deg=doas,
            cohernet_signals=cohernet_signals,
            M=8, N=TIME_WINDOW, fc=3.5e9, snr_db=snr,
            trials=3000,
            shrinkage_rho=0.0,   # keep 0.0 per your constraint
            denom_eps=1e-6,
            seed=123
        )
        results[snr] = summary
        print(f"\nSNR = {snr:>5.1f} dB")
        #print("  bias [deg] (per mode):", summary["empirical_bias_deg_wrapped_no_perm"])
        #print("  var  [deg^2] (per mode):", summary["empirical_var_deg2_wrapped_no_perm"])
        #print("Emprical bias is (θ̂ scale) : ", summary["bias"])
        #print("Emprical MSPE (θ̂ scale) : ", summary["mspe"])
        #print("Emprical Variance (θ̂ scale) : ", summary["emprical_var"])
        #print("Emprical bias is #1 (θ̂ scale) : ", summary["bias1"])
        #print("Emprical MSPE (θ̂ scale) #1  : ", summary["mspe1"])
        print("Emprical Variance (θ̂ scale) #1  : ", summary["emprical_var1"])
        print("Emprical Variance Mean (θ̂ scale) : ", summary["emprical_var_mean"])
        print("  pred [deg^2] (θ̂ scale):", summary["predicted_var_deg2_scale_theta_hat"])
        #print("  ratio empVar/pred  :", summary["ratio_empVar_over_pred_hat"])

    np.save('Results_file_uncertainty.npy', summary)
    # Optional: plot predicted variance vs SNR (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset  # mark_inset is optional

        K = len(doas)
        Y = np.array([results[snr]["predicted_var_deg2_scale_theta_hat"] for snr in snrs]).T  # (K x len(snrs))
        plt.figure()
        for i in range(K):
            plt.plot(snrs, Y[i, :], marker='o', label=f"Mode {i}")
        plt.xlabel("SNR [dB]")
        plt.ylabel("Predicted variance (deg²)\nEq. (58)")
        plt.title(f"ESPRIT Eq. (58) predicted DOA variance vs SNR\n(Δ=λ/2, M=8, N={TIME_WINDOW}")
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2) Empirical variance (deg², unbiased, with perm) vs SNR
        plt.figure()
        emprical_var_vs_snr = np.array([results[snr]["emprical_var1"] for snr in snrs]).T
        for k in range(K):
            plt.plot(snrs, emprical_var_vs_snr[k, :], marker='o', label=f"Mode {k}")
        #if use_logy:
        #    plt.yscale('log')
        plt.xlabel("SNR [dB]")
        plt.ylabel("Empirical variance (deg²)")
        plt.title("Empirical DOA variance vs SNR")
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 3) MSE (deg², with perm) vs SNR
        plt.figure()
        emprical_mspe_vs_snr = np.array([results[snr]["mspe1"] for snr in snrs]).T
        for k in range(K):
            plt.plot(snrs, emprical_mspe_vs_snr[k, :], marker='o', label=f"Mode {k}")
        #if use_logy:
        #    plt.yscale('log')
        plt.xlabel("SNR [dB]")
        plt.ylabel("Empirical MSE (deg²)")
        plt.title("Empirical DOA MSE vs SNR")
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 4) |Bias| (deg, with perm) vs SNR
        plt.figure()
        emprical_bias_vs_snr = np.array([results[snr]["bias1"] for snr in snrs]).T

        for k in range(K):
            plt.plot(snrs, np.abs(emprical_bias_vs_snr[k, :]), marker='o', label=f"Mode {k}")
        plt.xlabel("SNR [dB]")
        plt.ylabel("|Bias| (deg)")
        plt.title("Empirical |Bias| vs SNR")
        plt.grid(True, which='both')
        plt.legend()
        plt.tight_layout()
        plt.show()

        #Emprical Vs Predicated
        fig, ax = plt.subplots()

        for i in range(K):
            ax.plot(snrs, np.sqrt(Y)[i, :], linestyle='--', marker='o', label=f"Mode {i} Predicted")
            ax.plot(snrs, np.sqrt(emprical_var_vs_snr)[i, :], linestyle='--', marker='*', label=f"Mode {i} Empirical")

        # --- title + legend (legend sits below the title) ---
        ax.set_title(f"ESPRIT: Predicted vs Empirical DOA stddev vs SNR\n(Δ=λ/2, M=8, N={TIME_WINDOW})",
                     pad=30)  # push the title up a bit to make room

        # build a clean legend (avoid duplicates from any re-plotting)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5),
                  ncol=2, frameon=False)  # adjust ncol as you like

        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("Std-dev (deg)")
        ax.grid(True, which='both')
        plt.tight_layout()

        # ---------- Inset (magnified area over SNR [0, 20]) ----------
        snrs_arr = np.asarray(snrs)
        mask = (snrs_arr >= 0) & (snrs_arr <= 20)

        # Create the inset axes (size and location are adjustable)
        axins = inset_axes(ax, width="35%", height="35%", loc="upper right",   # (x, y) in axes coords (0–1)
     borderpad=1.0)

        # Re-plot the same curves into the inset
        for i in range(K):
            axins.plot(snrs, np.sqrt(Y)[i, :], linestyle='--', marker='o')
            axins.plot(snrs, np.sqrt(emprical_var_vs_snr)[i, :], linestyle='--', marker='*')


        # Limit inset to SNR 0–20
        axins.set_xlim(0, 20)

        # Set inset y-limits to the min/max of the data in that SNR window (with a small padding)
        ys_inset = np.concatenate([
            np.sqrt(Y)[:, mask].ravel(),
            np.sqrt(emprical_var_vs_snr)[:, mask].ravel()
        ]) if np.any(mask) else np.concatenate([np.sqrt(Y).ravel(), np.sqrt(emprical_var_vs_snr).ravel()])

        pad = 0.05 * (ys_inset.max() - ys_inset.min() + 1e-12)
        axins.set_ylim(ys_inset.min() - pad, ys_inset.max() + pad)

        # Optional cosmetics
        axins.tick_params(labelsize=6)
        ax.indicate_inset_zoom(axins, edgecolor="0.5")  # draws rectangle + connectors
        # If you're on an older Matplotlib, you can instead do:
        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=1)

        #plt.yscale('log')

        plt.show()

    except Exception as e:
        pass

