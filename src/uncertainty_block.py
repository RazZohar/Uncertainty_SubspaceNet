
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
    idx = np.argsort(evals)[-d_sources:].copy()
    E_s = evecs[:, idx]
    J1, J2 = build_overlapped_selectors(M, shift=shift)
    E_x, E_y = J1 @ E_s, J2 @ E_s
    F = np.linalg.pinv(E_x) @ E_y
    lam, V = np.linalg.eig(F)
    Q = np.linalg.inv(V)
    return {
        "E_s": E_s, "E_x": E_x, "E_y": E_y, "J1": J1, "J2": J2, "F": F,
        "lambda": lam, "V": V, "Q": Q,
        "evals": evals, "evecs": evecs, "sig_idx": idx
    }

# ================== 4th‑order moment (Eq. 63, Gaussian) ==================

def rcov_conj_gaussian_plugin(Rhat: np.ndarray, Ns: int) -> np.ndarray:
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
                    T3_H = np.tensordot(s_n_conj, T2_H, axes=(0, 0))  # (b2)
                    T3_T = np.tensordot(s_n_conj, T2_T, axes=(0, 1))  # (b2)
                    coeff_H = np.vdot(s_h, T3_H)
                    coeff_T = np.vdot(s_h, T3_T)
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
    """
    scale = eq58_scale_half_lambda(theta_i, eps)

    #mag = np.real(eq52_val) - np.real(eq53_val * (np.conj(lam_i)**2))
    mag = np.real(eq52_val) - np.real(eq53_val * (lam_i ** 2))

    #print(f'{scale=}, {mag=}, {eq52_val=}, {eq53_val=},  ')
    val = scale * mag
    return float(max(val, 0.0)) if clip_nonneg else float(val)
    """
    #var_lambda = np.real(eq52_val) - np.real(eq53_val * (lam_i ** 2))
    var_lambda = np.real(eq52_val) - np.real(eq53_val * (np.conj(lam_i) ** 2))

    # Yuen 96 Equation 58 scales the lambda variance by 1/2
    var_lambda = 0.5 * var_lambda

    # Safe clipping
    if clip_nonneg and var_lambda < 0:
        var_lambda = 1e-12

    # Derivative mapped to Yuen's domain (theta_rad now represents [0, pi])
    derivative_sq = (np.pi * np.sin(theta_i)) ** 2
    if derivative_sq < 1e-12:
        derivative_sq = 1e-12

    return var_lambda / derivative_sq

# ===================== Data generation (multi‑source) =====================


import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_sigma_vs_doa(doa_pred, sigma_pred, *, title="sigma_pred vs doa_pred"):
    """
    Scatter plot of sigma_pred against doa_pred.

    doa_pred, sigma_pred: array-like (numpy / torch / list), same shape when flattened.
    """
    doa = np.asarray(doa_pred, dtype=float).ravel()
    sig = np.asarray(sigma_pred, dtype=float).ravel()

    if doa.shape != sig.shape:
        raise ValueError(f"Shape mismatch after flatten: doa={doa.shape}, sigma={sig.shape}")

    plt.figure()
    plt.scatter(doa, sig)
    plt.xlabel("doa_pred [deg]")
    plt.ylabel("sigma_pred [deg]")
    plt.title(title)
    plt.show()


from scipy.optimize import linear_sum_assignment

class UncertaintyEstimation(nn.Module):
    def __init__(self, signal_shape):
        super().__init__()
        self.__signal_shape = signal_shape


    def doa_from_lam_deg(self, lam):
        # matches your esprit() sign convention
        s = np.clip(np.angle(lam) / np.pi, -1.0, 1.0)
        return np.rad2deg(-np.arcsin(s))

    def match_perm_to_external(self, doas_deg, lam):
        """
        Matches unordered ESPRIT roots (lam) to the Neural Network's ordered angles (doas_deg)
        by measuring distance directly on the Complex Unit Circle.
        """
        doas_deg = np.asarray(doas_deg)

        # 1. Convert the Network's physical angles into theoretical ESPRIT roots.
        # We use your system model's exact phase mapping: e^{-j * pi * sin(theta)}
        expected_lam = np.exp(-1j * np.pi * np.sin(np.deg2rad(doas_deg)))

        # 2. Normalize the actual ESPRIT roots to ensure they sit perfectly on the unit circle
        actual_lam = lam / np.abs(lam)

        # 3. Create a cost matrix based on Complex Euclidean Distance
        # This bypasses all arcsin coordinate ambiguities!
        cost_matrix = np.abs(expected_lam[:, None] - actual_lam[None, :])

        # 4. Hungarian algorithm to find the optimal 1-to-1 match
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        return col_ind, cost_matrix[row_ind, col_ind]

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

        Rhat = 0.5 * (Rhat + np.conj(Rhat).T)

        dsrc = len(doas_deg)

        # Collect across trials
        theta_hat_all = []  # list of (K,) in deg
        pred_hat_deg2_all = []  # list of (K,) predicted var using θ̂ in scale

        # 3) ESPRIT (overlapped, shift=1), use λ "as is"
        esp = esprit_overlapped(Rhat, d_sources=dsrc, shift=1)
        J1, J2, E_x = esp["J1"], esp["J2"], esp["E_x"]
        lam, V, Q = esp["lambda"], esp["V"], esp["Q"]

        # DOA from principal phase only (no |λ| projection, no phase unwrapping)
        #theta_es_deg = np.rad2deg(np.arcsin(np.clip(-np.angle(lam_u) / np.pi, -1, 1)))

        theta_hat_deg = doas_deg

        # 4) 4th‑order tensor (Gaussian plug‑in) and Eq.66 blocks
        Rcov_conj = rcov_conj_gaussian_plugin(Rhat, Ns=self.__signal_shape)
        alpha_hat = esp["evals"]
        S_hat = esp["evecs"]
        sig_idx = esp["sig_idx"]

        covHs_gh, covTs_gh = compute_delta_s_covariance_blocks_eq66(
            S_hat, alpha_hat, Rcov_conj, sig_idx, denom_eps=1e-6
        )

        perm_ext_to_int, doa_int = self.match_perm_to_external(theta_hat_deg, lam)
        lam_ext = lam[perm_ext_to_int]
        V_ext = V[:, perm_ext_to_int]
        Q_ext = Q[perm_ext_to_int, :]

        print("external doa:", theta_hat_deg)
        print("internal doa :", doa_int)
        print("matched doa  :", doa_int[perm_ext_to_int])

        matced_doas = doa_int[perm_ext_to_int]

        # 5) per‑mode Eq.52/53/58 (index‑aligned: i->i)
        pred_hat_deg2 = np.zeros(dsrc)
        for i in range(dsrc):
            v_i = V[:, i][:, None]
            q_i = Q[i, :][None, :]
            lam_i = lam[i]

            v_e = V_ext[:, i][:, None]
            q_e = Q_ext[i, :][None, :]
            lam_e = lam_ext[i]

            #alpha = (q_i @ v_i).item()
            #q_i = q_i / alpha

            eq52_i = compute_eq52_weighted(lam_i, q_i, E_x, J1, J2, covHs_gh, v_i)
            eq53_i = compute_eq53_weighted(lam_i, q_i, E_x, J1, J2, covTs_gh, v_i)


            var_hat = compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_i, eq53_i, np.deg2rad(theta_hat_deg[i] + 90.0),
                                                              clip_nonneg=True)



            eq52_e = compute_eq52_weighted(lam_e, q_e, E_x, J1, J2, covHs_gh, v_e)
            eq53_e = compute_eq53_weighted(lam_e, q_e, E_x, J1, J2, covTs_gh, v_e)

            ext_var_hat = compute_eq58_half_lambda_from_eq52_eq53(lam_e, eq52_e, eq53_e, np.deg2rad(theta_hat_deg[i] + 90.0),
                                                              clip_nonneg=True)


            print(f'when Using lam_ externel {ext_var_hat=} {lam_e=} {eq52_e=} {eq53_e=}, ')
            print(f'when Using lam_i {var_hat=} {lam_i=} {eq52_i=} {eq53_i=}, ')


            pred_hat_deg2[i] = var_hat * ((180 / np.pi) ** 2)
            #print(f'when Using lam_i {var_hat=} {lam_i=} {eq52_i=} {eq53_i=}, ')

        theta_hat_all.append(theta_hat_deg)
        pred_hat_deg2_all.append(pred_hat_deg2)

        pred_hat_m = np.mean(np.vstack(pred_hat_deg2_all), axis=0)  # (K,)
        #print(f'{pred_hat_m=}')

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
        return torch.sqrt(uncertainty)
