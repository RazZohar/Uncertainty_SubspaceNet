
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
import scipy
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
    lam, Q, V = scipy.linalg.eig(F, left=True, right=True)
    return {
        "E_s": E_s, "E_x": E_x, "E_y": E_y, "J1": J1, "J2": J2, "F": F,
        "lambda": lam, "V": V, "Q": Q,
        "evals": evals, "evecs": evecs, "sig_idx": idx
    }

# ================== 4th‑order moment (Eq. 63, Gaussian) ==================

def rcov_conj_gaussian_plugin(Rhat: np.ndarray, Ns: int) -> np.ndarray:
    #return (Rhat[:, None, :, None] * Rhat[None, :, None, :]) / Ns

    term1 = Rhat[:, None, :, None] * np.conj(Rhat)[None, :, None, :]
    term2 = Rhat[:, None, None, :] * np.conj(Rhat)[None, :, :, None]
    return (term1 + term2) / Ns

# =============== Eq. 66: eigenvector perturbation blocks ===============

def compute_delta_s_covariance_blocks_eq66(
    S_full: np.ndarray,
    alpha: np.ndarray,
    Rcov_conj: np.ndarray,
    signal_indices,
    denom_eps: float = 1e-12,
    use_full_sums: bool = True,   # True = literal Eq. (62)/(66)
):
    """
    Returns covHs_gh, covTs_gh of shape (d, d, M, M).

    If use_full_sums=False:
        sums only over noise eigenvectors (common reduced form in subspace perturbation use).
    If use_full_sums=True:
        uses the literal paper sums l != g and n != h.
    """
    S_full = np.asarray(S_full)
    alpha = np.asarray(alpha)
    Rcov_conj = np.asarray(Rcov_conj)

    M = S_full.shape[0]
    sig_idx = np.asarray(signal_indices)
    d = sig_idx.size

    # Eq. (66) note: E{ΔR[a1,a2] ΔR[b1,b2]} = E{ΔR[a1,a2] ΔR*[b2,b1]}
    Rcov_unconj = np.swapaxes(Rcov_conj, 2, 3)

    covHs_gh = np.zeros((d, d, M, M), dtype=complex)
    covTs_gh = np.zeros((d, d, M, M), dtype=complex)

    noise_idx = np.setdiff1d(np.arange(M), sig_idx)

    for gi, g in enumerate(sig_idx):
        s_g = S_full[:, g]

        l_indices = np.setdiff1d(np.arange(M), [g]) if use_full_sums else noise_idx

        for hi, h in enumerate(sig_idx):
            s_h = S_full[:, h]

            n_indices = np.setdiff1d(np.arange(M), [h]) if use_full_sums else noise_idx

            accum_H = np.zeros((M, M), dtype=complex)
            accum_T = np.zeros((M, M), dtype=complex)

            for l in l_indices:
                s_l = S_full[:, l]

                # common first two contractions: s_l^* , s_g
                T1_H = np.tensordot(s_l.conj(), Rcov_conj, axes=(0, 0))   # [a2,b1,b2]
                T1_T = np.tensordot(s_l.conj(), Rcov_unconj, axes=(0, 0)) # [a2,b1,b2]

                T2_H = np.tensordot(s_g, T1_H, axes=(0, 0))               # [b1,b2]
                T2_T = np.tensordot(s_g, T1_T, axes=(0, 0))               # [b1,b2]

                for n in n_indices:
                    s_n = S_full[:, n]

                    # Eq. (62): ... s_{n,b1} s^*_{h,b2}
                    T3_H = np.tensordot(s_n, T2_H, axes=(0, 0))           # [b2]
                    coeff_H = np.vdot(s_h, T3_H)                          # sum_b2 s_h^*[b2] * T3_H[b2]

                    # Eq. (66): ... s^*_{n,b1} s_{h,b2}
                    T3_T = np.tensordot(s_n.conj(), T2_T, axes=(0, 0))    # [b2]
                    coeff_T = np.dot(T3_T, s_h)                           # no conjugation on s_h

                    denom = (alpha[g] - alpha[l]) * (alpha[h] - alpha[n])

                    if abs(denom) < denom_eps:
                        continue

                    accum_H += (coeff_H / denom) * np.outer(s_l, s_n.conj())  # s_l s_n^H
                    accum_T += (coeff_T / denom) * np.outer(s_l, s_n)         # s_l s_n^T

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

def enforce_psd_onto_matrix(matrix):
    """
    I use Hermittian and then Diagonal Loading
    :param matrix:
    :return: PSD matrix from the given input matrix
    """
    #M = (matrix + matrix.conj().T) / 2
    M = matrix
    eigenvalues = np.linalg.eigvals(M)
    min_eig = np.min(eigenvalues)

    # 3. Calculate the required shift
    epsilon = 1e-6
    shift = max(0, -min_eig) + epsilon

    # 4. Apply diagonal loading
    M_psd = M + shift * np.eye(M.shape[0])

    return M_psd

def compute_eq52_weighted(lam_i, q_i, E_x, W1, W2, covHs_gh, v_i):
    E_x_pinv = np.linalg.pinv(E_x)
    middle = _assemble_middle_from_v(v_i, covHs_gh, conjugate_h=True)

    if np.min(np.linalg.eigvals(middle)) < 0:
        print(f'52 middle {np.linalg.eigvals(middle)=} result non PSD matrix')
    left   = q_i @ E_x_pinv @ (W1 - np.conj(lam_i) * W2)                 # (1 x M)
    right  = (W1 - np.conj(lam_i) * W2).conj().T @ E_x_pinv.conj().T @ q_i.conj().T  # (M x 1)
    return (left @ middle @ right).item()

def compute_eq53_weighted(lam_i, q_i, E_x, W1, W2, covTs_gh, v_i):
    E_x_pinv = np.linalg.pinv(E_x)
    middle = _assemble_middle_from_v(v_i, covTs_gh, conjugate_h=False)
    """if np.min(np.linalg.eigvals(middle)) < 0:
        print(f'53 middle {np.linalg.eigvals(middle)=} result non PSD matrix')
        middle = enforce_psd_onto_matrix(middle)
        print(f'{np.linalg.eigvals(middle)=} , {np.min(np.linalg.eigvals(middle))=}')
    """
    left   = q_i @ E_x_pinv @ (W2 - lam_i * W1)
    right  = (W2 - lam_i * W1).T @ E_x_pinv.T @ q_i.T
    return (left @ middle @ right).item()
# ================= Eq. 58 with Δ = λ/2 simplified =================

def eq58_scale_half_lambda(theta_i: float, eps: float = 1e-8) -> float:
    c2 = max(np.cos(theta_i + np.pi/2)**2, eps)  # guard endfire
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
    """

    scale = eq58_scale_half_lambda(theta_i, eps)

    # Calculate the true magnitude squared of the empirical eigenvalue
    mag_sq = np.abs(lam_i) ** 2

    # Correct normalizations derived from Var(Im(d_lambda / lambda))
    term1 = np.real(eq52_val)# / mag_sq

    # eq53_val / lam_i^2 is mathematically equivalent to (eq53_val * (lam_i^*)^2) / |lam_i|^4
    term2 = np.real(eq53_val * (np.conj(lam_i) ** 2))# / (mag_sq ** 2)

    var_lambda = term1 - term2

    val = scale * var_lambda
    return float(max(val, 0.0)) if clip_nonneg else float(val)

# ===================== Data generation (multi‑source) =====================


import numpy as np
import matplotlib.pyplot as plt
import itertools

import numpy as np
import matplotlib.pyplot as plt
import math

def plot_sigma_vs_doa(doa_pred, sigma_pred, *, title="sigma_pred vs doa_pred"):
    """
    Plot sigma_pred against doa_pred with one polar subplot per target.

    Expected shape:
        doa_pred   : [Batch, num_targets]
        sigma_pred : [Batch, num_targets]
    """
    doa = np.asarray(doa_pred, dtype=float)
    sig = np.asarray(sigma_pred, dtype=float)

    if doa.shape != sig.shape:
        raise ValueError(f"Shape mismatch: doa={doa.shape}, sigma={sig.shape}")

    if doa.ndim != 2:
        raise ValueError(f"Expected shape [Batch, num_targets], got {doa.shape}")

    batch_size, num_targets = doa.shape

    ncols = math.ceil(math.sqrt(num_targets))
    nrows = math.ceil(num_targets / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        subplot_kw={'projection': 'polar'},
        layout='constrained',
        figsize=(4 * ncols, 4 * nrows)
    )

    axes = np.array(axes).reshape(-1)

    for target_idx in range(num_targets):
        ax = axes[target_idx]

        doa_target = doa[:, target_idx]
        sig_target = sig[:, target_idx]

        ax.scatter(doa_target + 90.0, sig_target, marker='o', label=f'Target {target_idx+1}')
        ax.set_title(f"Target {target_idx+1}", va='bottom')
        ax.set_rlabel_position(-30.0)
        ax.grid(True)
        ax.set_rorigin(0)
        ax.set_rmax(min(float(np.max(sig_target) * 1.20), 90.0))
        ax.set_thetamin(np.min(doa_target + 90.0) - 10)
        ax.set_thetamax(np.max(doa_target + 90.0) + 10)
        ax.legend()

    # Hide unused subplots
    for i in range(num_targets, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title)
    plt.show()

from scipy.optimize import linear_sum_assignment

class UncertaintyEstimation(nn.Module):
    def __init__(self, signal_shape):
        super().__init__()
        self.__signal_shape = signal_shape
        self.__subarray_shift = 1

    def set_subarray_shift(self, subarray_shift: int):
        self.__subarray_shift = subarray_shift

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

        #self.__subarray_shift = 4
        # 3) ESPRIT (overlapped, shift=1), use λ "as is"
        esp = esprit_overlapped(Rhat, d_sources=dsrc, shift=self.__subarray_shift)
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

        """for g in range(dsrc):
            covHs_gh[g, g] = 0.5 * (covHs_gh[g, g] + covHs_gh[g, g].conj().T)

        for g in range(dsrc):
            for h in range(g + 1, dsrc):
                A = 0.5 * (covHs_gh[g, h] + covHs_gh[h, g].conj().T)
                covHs_gh[g, h] = A
                covHs_gh[h, g] = A.conj().T
        """
        #lam_u = lam / np.maximum(np.abs(lam), 1e-12)
        lam = lam / np.abs(lam)
        perm_ext_to_int, doa_int = self.match_perm_to_external(theta_hat_deg, lam)
        lam_ext = lam[perm_ext_to_int]
        V_ext = V[:, perm_ext_to_int]
        Q_ext = Q[:, perm_ext_to_int]

        print("external doa:", theta_hat_deg)
        #print("internal doa :", doa_int)
        print("matched doa  :", self.doa_from_lam_deg(lam_ext))

        matced_doas =  self.doa_from_lam_deg(lam_ext)

        # 5) per‑mode Eq.52/53/58 (index‑aligned: i->i)
        pred_hat_deg2 = np.zeros(dsrc)
        for i in range(dsrc):
            """
            v_i = V[:, i][:, None]
            q_i = Q[:, i].conj()[None, :]
            q_i = q_i / (q_i @ v_i)
            lam_i = lam[i]
            """
            v_i = V_ext[:, i][:, None]
            q_i = Q_ext[:, i].conj()[None, :]
            q_i = q_i / (q_i @ v_i)
            lam_i = lam_ext[i]

            #v_e = V_ext[:, i][:, None]
            #q_e = Q_ext[:, i].conj()[None, :]
            #q_e = q_e / (q_e @ v_e)
            #lam_e = lam_ext[i]

            eq52_i = compute_eq52_weighted(lam_i, q_i, E_x, J1, J2, covHs_gh, v_i)
            eq53_i = compute_eq53_weighted(lam_i, q_i, E_x, J1, J2, covTs_gh, v_i)


            var_hat = compute_eq58_half_lambda_from_eq52_eq53(lam_i, eq52_i, eq53_i, np.deg2rad(theta_hat_deg[i]),
                                                              clip_nonneg=True)



            #eq52_e = compute_eq52_weighted(lam_e, q_e, E_x, J1, J2, covHs_gh, v_e)
            #eq53_e = compute_eq53_weighted(lam_e, q_e, E_x, J1, J2, covTs_gh, v_e)

            #ext_var_hat = compute_eq58_half_lambda_from_eq52_eq53(lam_e, eq52_e, eq53_e, np.deg2rad(theta_hat_deg[i]),
            #                                                  clip_nonneg=True)


            #print(f'when Using lam_ externel {ext_var_hat=} {lam_e=} {eq52_e=} {eq53_e=}, {v_e=}, {q_e=}')
            pred_hat_deg2[i] = var_hat * ((180 / np.pi) ** 2)
            print(f'when Using lam_i {np.sqrt(pred_hat_deg2[i])=} {pred_hat_deg2[i]=} {var_hat=} {lam_i=} {eq52_i=} {eq53_i=}, {v_i=}, {q_i=}')



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

        plot_sigma_vs_doa(doas_deg, torch.sqrt(uncertainty))
        return torch.sqrt(uncertainty)
