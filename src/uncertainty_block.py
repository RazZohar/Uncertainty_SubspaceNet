# ---------------- uncertainty_block.py ----------------
"""
Batched PyTorch implementation of the theoretical ESPRIT uncertainty block.



Expected interface:
    sigma_deg = UncertaintyEstimation(signal_shape=T)(doas_deg, Rx)

where:
    doas_deg : [B, M]        real tensor, degrees, same convention as the network output
    Rx       : [B, N, N]     complex covariance matrix
    output   : [B, M]        sigma / standard deviation in degrees

"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Optional

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _as_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.complex64, torch.complex128):
        return dtype
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def _real_dtype_from_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    return torch.float32


def build_overlapped_selectors_torch(
    n_antennas: int,
    shift: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build overlapped ESPRIT selectors J1,J2.

    J1 selects sensors [0, ..., N-shift-1]
    J2 selects sensors [shift, ..., N-1]
    """
    rows = n_antennas - shift
    if rows <= 0:
        raise ValueError(f"Invalid shift={shift} for n_antennas={n_antennas}")

    J1 = torch.zeros(rows, n_antennas, device=device, dtype=dtype)
    J2 = torch.zeros(rows, n_antennas, device=device, dtype=dtype)

    idx = torch.arange(rows, device=device)
    J1[idx, idx] = 1.0
    J2[idx, idx + shift] = 1.0

    return J1, J2


def rcov_conj_gaussian_plugin_torch(Rhat: torch.Tensor, ns: int | float) -> torch.Tensor:
    """
    Batched Gaussian plug-in covariance of covariance entries.

    Matches the NumPy code:

        term1 = Rhat[:, None, :, None] * conj(Rhat)[None, :, None, :]
        term2 = Rhat[:, None, None, :] * conj(Rhat)[None, :, :, None]
        Rcov_conj = (term1 + term2) / Ns

    Batched shape:
        Rhat       : [B, N, N]
        Rcov_conj  : [B, N, N, N, N]

    Axis meaning:
        Rcov_conj[b, a1, a2, b1, b2]
    """
    if Rhat.ndim != 3:
        raise ValueError(f"Rhat must be [B,N,N], got {tuple(Rhat.shape)}")

    term1 = Rhat[:, :, None, :, None] * Rhat.conj()[:, None, :, None, :]
    term2 = Rhat[:, :, None, None, :] * Rhat.conj()[:, None, :, :, None]
    return (term1 + term2) / float(ns)


# -----------------------------------------------------------------------------
# Batched ESPRIT
# -----------------------------------------------------------------------------

def batched_esprit_overlapped_torch(
    Rhat: torch.Tensor,
    d_sources: int,
    shift: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    Batched classical ESPRIT with overlapped subarrays.

    Rhat:
        [B, N, N] complex Hermitian covariance matrices

    Returns:
        E_s, E_x, E_y, J1, J2, F, lambda, V, evals, evecs, sig_idx
    """
    if Rhat.ndim != 3:
        raise ValueError(f"Rhat must be [B,N,N], got {tuple(Rhat.shape)}")

    B, N, N2 = Rhat.shape
    if N != N2:
        raise ValueError(f"Rhat must be square, got {tuple(Rhat.shape)}")
    if d_sources <= 0 or d_sources >= N:
        raise ValueError(f"d_sources must satisfy 0 < d_sources < N, got d_sources={d_sources}, N={N}")

    device = Rhat.device
    dtype = Rhat.dtype

    # torch.linalg.eigh returns ascending eigenvalues and matching eigenvectors.
    evals, evecs = torch.linalg.eigh(Rhat)  # evals: [B,N], evecs: [B,N,N]

    # Signal subspace: eigenvectors of largest d eigenvalues.
    sig_idx = torch.arange(N - d_sources, N, device=device)
    E_s = evecs[:, :, -d_sources:]  # [B,N,d]

    J1, J2 = build_overlapped_selectors_torch(
        n_antennas=N,
        shift=shift,
        device=device,
        dtype=dtype,
    )

    E_x = torch.einsum("rn,bnd->brd", J1, E_s)  # [B,rows,d]
    E_y = torch.einsum("rn,bnd->brd", J2, E_s)  # [B,rows,d]

    F = torch.linalg.pinv(E_x) @ E_y             # [B,d,d]
    lam, V = torch.linalg.eig(F)                 # lam: [B,d], V: [B,d,d]

    return {
        "E_s": E_s,
        "E_x": E_x,
        "E_y": E_y,
        "J1": J1,
        "J2": J2,
        "F": F,
        "lambda": lam,
        "V": V,
        "evals": evals,
        "evecs": evecs,
        "sig_idx": sig_idx,
    }


# -----------------------------------------------------------------------------
# Batched permutation matching
# -----------------------------------------------------------------------------

def _permutation_tensor(d: int, device: torch.device) -> torch.Tensor:
    """
    All permutations of range(d). Fine for small number of sources M=1..4.
    """
    return torch.tensor(
        list(itertools.permutations(range(d))),
        device=device,
        dtype=torch.long,
    )


def batched_match_perm_to_external_torch(
    doas_deg: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    """
    Torch replacement for scipy.optimize.linear_sum_assignment.

    doas_deg:
        [B,d] network/external DOAs in degrees

    lam:
        [B,d] ESPRIT roots/eigenvalues

    Returns:
        perm: [B,d]
        perm[b,i] is the ESPRIT/internal eigenvalue index matched to external DOA i.

    This brute-force matcher is very fast for d=1,2,3,4 and avoids CPU/SciPy.
    """
    if doas_deg.ndim != 2:
        raise ValueError(f"doas_deg must be [B,d], got {tuple(doas_deg.shape)}")
    if lam.ndim != 2:
        raise ValueError(f"lam must be [B,d], got {tuple(lam.shape)}")

    B, d = doas_deg.shape
    if lam.shape != (B, d):
        raise ValueError(f"Shape mismatch doas_deg={tuple(doas_deg.shape)}, lam={tuple(lam.shape)}")

    device = doas_deg.device
    real_dtype = doas_deg.dtype

    # Same convention as your old NumPy code:
    # expected_lam = exp(-j*pi*sin(theta_deg))
    expected_lam = torch.exp(
        -1j * torch.tensor(math.pi, device=device, dtype=real_dtype) * torch.sin(torch.deg2rad(doas_deg))
    )  # [B,d]

    actual_lam = lam / lam.abs().clamp_min(1e-12)

    # cost[b, external_i, internal_j]
    cost = (expected_lam[:, :, None] - actual_lam[:, None, :]).abs()

    if d == 1:
        return torch.zeros(B, 1, device=device, dtype=torch.long)

    perms = _permutation_tensor(d, device=device)  # [P,d]
    P = perms.shape[0]

    # selected_cost[b,p,i] = cost[b,i,perms[p,i]]
    cost_expanded = cost[:, None, :, :].expand(B, P, d, d)  # [B,P,d,d]
    gather_index = perms[None, :, :, None].expand(B, P, d, 1)  # [B,P,d,1]

    selected_cost = torch.gather(
        cost_expanded,
        dim=3,
        index=gather_index,
    ).squeeze(-1)  # [B,P,d]

    scores = selected_cost.sum(dim=-1)  # [B,P]
    best = torch.argmin(scores, dim=1)  # [B]

    return perms[best]  # [B,d]


def batched_doa_from_lam_deg_torch(lam: torch.Tensor) -> torch.Tensor:
    """
    DOA from ESPRIT root using the same sign convention as your old helper:

        s = angle(lam) / pi
        doa = -arcsin(s)

    Returns degrees.
    """
    s = torch.clamp(torch.angle(lam) / math.pi, -1.0, 1.0)
    return torch.rad2deg(-torch.arcsin(s))


# -----------------------------------------------------------------------------
# Eq. 66 blocks
# -----------------------------------------------------------------------------

def compute_delta_s_covariance_blocks_eq66_torch(
    S_full: torch.Tensor,
    alpha: torch.Tensor,
    Rcov_conj: torch.Tensor,
    d_sources: int,
    denom_eps: float = 1e-6,
    use_full_sums: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched torch version of compute_delta_s_covariance_blocks_eq66.

    Inputs:
        S_full:
            [B,N,N] full eigenvector matrix from torch.linalg.eigh

        alpha:
            [B,N] eigenvalues from torch.linalg.eigh

        Rcov_conj:
            [B,N,N,N,N]

        d_sources:
            number of sources

    Returns:
        covHs_gh:
            [B,d,d,N,N]

        covTs_gh:
            [B,d,d,N,N]
    """
    if S_full.ndim != 3:
        raise ValueError(f"S_full must be [B,N,N], got {tuple(S_full.shape)}")
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be [B,N], got {tuple(alpha.shape)}")
    if Rcov_conj.ndim != 5:
        raise ValueError(f"Rcov_conj must be [B,N,N,N,N], got {tuple(Rcov_conj.shape)}")

    B, N, N2 = S_full.shape
    if N != N2:
        raise ValueError(f"S_full must be square, got {tuple(S_full.shape)}")
    if alpha.shape != (B, N):
        raise ValueError(f"alpha shape mismatch: alpha={tuple(alpha.shape)}, expected {(B, N)}")
    if Rcov_conj.shape != (B, N, N, N, N):
        raise ValueError(f"Rcov_conj shape mismatch: {tuple(Rcov_conj.shape)}")

    device = S_full.device
    dtype = S_full.dtype

    sig_idx = list(range(N - d_sources, N))
    noise_idx = list(range(0, N - d_sources))

    # Same as np.swapaxes(Rcov_conj, 2, 3) on the non-batched tensor.
    # Batched tensor axes are [B,a1,a2,b1,b2], so swap b1/b2 -> dims 3 and 4.
    Rcov_unconj = Rcov_conj.transpose(3, 4)

    covHs = torch.zeros(B, d_sources, d_sources, N, N, device=device, dtype=dtype)
    covTs = torch.zeros_like(covHs)

    for gi, g in enumerate(sig_idx):
        s_g = S_full[:, :, g]  # [B,N]

        l_indices = [idx for idx in range(N) if idx != g] if use_full_sums else noise_idx

        for hi, h in enumerate(sig_idx):
            s_h = S_full[:, :, h]  # [B,N]

            n_indices = [idx for idx in range(N) if idx != h] if use_full_sums else noise_idx

            accum_H = torch.zeros(B, N, N, device=device, dtype=dtype)
            accum_T = torch.zeros(B, N, N, device=device, dtype=dtype)

            for l in l_indices:
                s_l = S_full[:, :, l]  # [B,N]

                for n in n_indices:
                    s_n = S_full[:, :, n]  # [B,N]

                    # Equivalent to:
                    # sum_{a1,a2,b1,b2} conj(s_l[a1]) * s_g[a2] *
                    #                      s_n[b1] * conj(s_h[b2]) *
                    #                      Rcov_conj[a1,a2,b1,b2]
                    coeff_H = torch.einsum(
                        "bp,bq,br,bs,bpqrs->b",
                        s_l.conj(),
                        s_g,
                        s_n,
                        s_h.conj(),
                        Rcov_conj,
                    )

                    # Equivalent to the old covTs branch with Rcov_unconj.
                    coeff_T = torch.einsum(
                        "bp,bq,br,bs,bpqrs->b",
                        s_l.conj(),
                        s_g,
                        s_n.conj(),
                        s_h,
                        Rcov_unconj,
                    )

                    denom = (alpha[:, g] - alpha[:, l]) * (alpha[:, h] - alpha[:, n])
                    safe = denom.abs() > denom_eps
                    safe_denom = torch.where(safe, denom, torch.ones_like(denom))

                    coeff_H = torch.where(safe, coeff_H / safe_denom, torch.zeros_like(coeff_H))
                    coeff_T = torch.where(safe, coeff_T / safe_denom, torch.zeros_like(coeff_T))

                    outer_H = s_l[:, :, None] * s_n.conj()[:, None, :]  # [B,N,N]
                    outer_T = s_l[:, :, None] * s_n[:, None, :]         # [B,N,N]

                    accum_H = accum_H + coeff_H[:, None, None] * outer_H
                    accum_T = accum_T + coeff_T[:, None, None] * outer_T

            covHs[:, gi, hi] = accum_H
            covTs[:, gi, hi] = accum_T

    return covHs, covTs


# -----------------------------------------------------------------------------
# Eq. 52 / Eq. 53 / Eq. 58
# -----------------------------------------------------------------------------

def compute_eq52_eq53_all_modes_torch(
    lam_ext: torch.Tensor,
    V_ext: torch.Tensor,
    E_x: torch.Tensor,
    J1: torch.Tensor,
    J2: torch.Tensor,
    covHs: torch.Tensor,
    covTs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Eq.52/Eq.53 for all modes.

    Inputs:
        lam_ext:
            [B,d] matched ESPRIT eigenvalues

        V_ext:
            [B,d,d], columns are right eigenvectors matched to external DOA order

        E_x:
            [B,rows,d]

        J1,J2:
            [rows,N]

        covHs,covTs:
            [B,d,d,N,N]

    Returns:
        eq52:
            [B,d] complex

        eq53:
            [B,d] complex
    """
    if lam_ext.ndim != 2:
        raise ValueError(f"lam_ext must be [B,d], got {tuple(lam_ext.shape)}")
    if V_ext.ndim != 3:
        raise ValueError(f"V_ext must be [B,d,d], got {tuple(V_ext.shape)}")

    B, d = lam_ext.shape

    # E_x_pinv: [B,d,rows]
    E_x_pinv = torch.linalg.pinv(E_x)

    # Rows of inv(V) are left eigenvector rows satisfying q_i @ V[:,j] = delta_ij.
    # This matches the old use of q_i normalized so q_i @ v_i = 1.
    Q_rows = torch.linalg.inv(V_ext)  # [B,d,d]

    # middle52_i = sum_{g,h} v_i[g] conj(v_i[h]) covHs[g,h]
    # middle53_i = sum_{g,h} v_i[g]      v_i[h]  covTs[g,h]
    W52 = torch.einsum("bgi,bhi->bigh", V_ext, V_ext.conj())  # [B,i,g,h]
    W53 = torch.einsum("bgi,bhi->bigh", V_ext, V_ext)         # [B,i,g,h]

    middle52 = torch.einsum("bigh,bghmn->bimn", W52, covHs)   # [B,d,N,N]
    middle53 = torch.einsum("bigh,bghmn->bimn", W53, covTs)   # [B,d,N,N]

    # Eq.52:
    # q_i @ pinv(E_x) @ (J1 - conj(lambda_i) J2)
    A52 = J1[None, None, :, :] - lam_ext.conj()[:, :, None, None] * J2[None, None, :, :]

    # Eq.53:
    # q_i @ pinv(E_x) @ (J2 - lambda_i J1)
    A53 = J2[None, None, :, :] - lam_ext[:, :, None, None] * J1[None, None, :, :]

    left52 = torch.einsum("bid,bdr,birn->bin", Q_rows, E_x_pinv, A52)  # [B,d,N]
    left53 = torch.einsum("bid,bdr,birn->bin", Q_rows, E_x_pinv, A53)  # [B,d,N]

    eq52 = torch.einsum("bin,binm,bim->bi", left52, middle52, left52.conj())
    eq53 = torch.einsum("bin,binm,bim->bi", left53, middle53, left53)

    return eq52, eq53


def compute_eq58_half_lambda_torch(
    lam_ext: torch.Tensor,
    eq52: torch.Tensor,
    eq53: torch.Tensor,
    theta_deg: torch.Tensor,
    eps: float = 1e-8,
    max_var_deg2: float = 8100.0,
) -> torch.Tensor:
    """
    Torch version matching the active NumPy Eq.58 branch.

    Old code did:
        var_hat = compute_eq58_half_lambda_from_eq52_eq53(
            lam_i, eq52_i, eq53_i, deg2rad(theta_hat_deg[i] + 90)
        )

    and inside:
        eq58_scale_half_lambda(theta_i) uses cos(theta_i + pi/2)^2

    Therefore:
        cos((theta_deg + 90deg) + 90deg)^2
        = cos(theta_deg + 180deg)^2
        = cos(theta_deg)^2

    So this implementation directly uses cos(theta_deg)^2.

    Also matches the active branch:
        term1 = real(eq52)
        term2 = real(eq53 * conj(lam)^2)
        var_lambda = term1 - term2
    """
    real_dtype = theta_deg.dtype
    theta = torch.deg2rad(theta_deg)

    c2 = torch.cos(theta).pow(2).clamp_min(eps)
    scale = 1.0 / (2.0 * math.pi**2 * c2)

    var_lambda = eq52.real - (eq53 * (lam_ext.conj() ** 2)).real
    var_rad2 = torch.clamp(scale * var_lambda, min=0.0)

    deg_per_rad = torch.tensor(180.0 / math.pi, device=theta_deg.device, dtype=real_dtype)
    var_deg2 = var_rad2 * deg_per_rad.pow(2)

    return torch.clamp(var_deg2, min=0.0, max=max_var_deg2)


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
    doa_pred = doa_pred.detach().numpy()
    sigma_pred = sigma_pred.detach().numpy()

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

# -----------------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------------

class UncertaintyEstimation(nn.Module):
    """
    Batched GPU/torch UncertaintyEstimation.

    Args:
        signal_shape:
            Number of snapshots Ns/T used in the Gaussian plug-in covariance.

        subarray_shift:
            ESPRIT shift. Your current old block defaults to 1.

        complex_dtype:
            Optional forced complex dtype. Use torch.complex64 for speed,
            torch.complex128 for debugging/numerical comparison.

        chunk_size:
            Optional chunk size over the batch dimension to reduce memory.
            Example: chunk_size=128. This still avoids per-sample CPU/SciPy loops.

        use_full_sums:
            Matches the old Eq.66 implementation when True.
    """

    def __init__(
        self,
        signal_shape: int | float,
        subarray_shift: int = 1,
        complex_dtype: Optional[torch.dtype] = None,
        chunk_size: Optional[int] = None,
        denom_eps: float = 1e-6,
        use_full_sums: bool = True,
        max_var_deg2: float = 8100.0,
    ):
        super().__init__()
        self.signal_shape = signal_shape
        self.subarray_shift = int(subarray_shift)
        self.complex_dtype = complex_dtype
        self.chunk_size = chunk_size
        self.denom_eps = float(denom_eps)
        self.use_full_sums = bool(use_full_sums)
        self.max_var_deg2 = float(max_var_deg2)

    def set_subarray_shift(self, subarray_shift: int):
        self.subarray_shift = int(subarray_shift)

    def forward(self, doas_deg: torch.Tensor, Rx: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        """
        Return sigma/std in degrees.

        doas_deg:
            [B,M] in degrees

        Rx:
            [B,N,N] complex covariance matrix

        Returns:
            If return_cov_diag is False:
                sigma_deg:
                    [B,M], standard deviation in degrees

            If return_cov_diag is True:
                sigma_deg, cov_diag_deg2:
                    both [B,M], where cov_diag_deg2 is the matched covariance
                    diagonal / variance in deg^2
        """
        if doas_deg.ndim != 2:
            raise ValueError(f"doas_deg must be [B,M], got {tuple(doas_deg.shape)}")
        if Rx.ndim != 3:
            raise ValueError(f"Rx must be [B,N,N], got {tuple(Rx.shape)}")
        if Rx.shape[0] != doas_deg.shape[0]:
            raise ValueError(f"Batch mismatch doas_deg={tuple(doas_deg.shape)}, Rx={tuple(Rx.shape)}")

        B = doas_deg.shape[0]
        if self.chunk_size is not None and B > self.chunk_size:
            sigma_chunks = []
            cov_chunks = []

            for start in range(0, B, self.chunk_size):
                end = min(start + self.chunk_size, B)
                out = self._forward_impl(doas_deg[start:end], Rx[start:end])

                if self.return_cov_diag:
                    sigma_chunk, cov_chunk = out
                    sigma_chunks.append(sigma_chunk)
                    cov_chunks.append(cov_chunk)
                else:
                    sigma_chunks.append(out)

            if self.return_cov_diag:
                return torch.cat(sigma_chunks, dim=0), torch.cat(cov_chunks, dim=0)

            return torch.cat(sigma_chunks, dim=0)

        return self._forward_impl(doas_deg, Rx)

    def _forward_impl(self, doas_deg: torch.Tensor, Rx: torch.Tensor) -> tuple(torch.Tensor, torch.Tensor):
        B, d_sources = doas_deg.shape

        complex_dtype = self.complex_dtype or _as_complex_dtype(Rx.dtype)
        real_dtype = _real_dtype_from_complex(complex_dtype)

        Rx = Rx.to(dtype=complex_dtype)
        doas_deg = doas_deg.to(device=Rx.device, dtype=real_dtype)

        # Hermitian symmetrization, same as old NumPy code.
        Rhat = 0.5 * (Rx + Rx.conj().transpose(-1, -2))

        esp = batched_esprit_overlapped_torch(
            Rhat,
            d_sources=d_sources,
            shift=self.subarray_shift,
        )

        lam = esp["lambda"]                          # [B,d]
        V = esp["V"]                                  # [B,d,d]
        E_x = esp["E_x"]                              # [B,rows,d]
        J1 = esp["J1"]                                # [rows,N]
        J2 = esp["J2"]                                # [rows,N]
        alpha_hat = esp["evals"]                      # [B,N]
        S_hat = esp["evecs"]                          # [B,N,N]

        # Old code normalized roots to unit circle before matching.
        lam = lam / lam.abs().clamp_min(1e-12)

        # Match ESPRIT modes to the network/external DOA order.
        #
        # perm_ext_to_int[b, i] is the internal ESPRIT mode corresponding to
        # external/network DOA doas_deg[b, i].
        #
        # After gathering, lam_ext[:, i] and V_ext[:, :, i] are aligned with
        # doas_deg[:, i]. Therefore var_deg2 / cov_diag_deg2 is also returned
        # in the external DOA order, not the arbitrary ESPRIT eigenvalue order.
        perm_ext_to_int = batched_match_perm_to_external_torch(doas_deg, lam)

        lam_ext = torch.gather(lam, dim=1, index=perm_ext_to_int)  # [B,d]

        # Gather columns of V by matched permutation.
        V_ext = torch.gather(
            V,
            dim=2,
            index=perm_ext_to_int[:, None, :].expand(-1, d_sources, -1),
        )  # [B,d,d]

        Rcov_conj = rcov_conj_gaussian_plugin_torch(
            Rhat,
            ns=self.signal_shape,
        )  # [B,N,N,N,N]

        covHs, covTs = compute_delta_s_covariance_blocks_eq66_torch(
            S_full=S_hat,
            alpha=alpha_hat,
            Rcov_conj=Rcov_conj,
            d_sources=d_sources,
            denom_eps=self.denom_eps,
            use_full_sums=self.use_full_sums,
        )

        eq52, eq53 = compute_eq52_eq53_all_modes_torch(
            lam_ext=lam_ext,
            V_ext=V_ext,
            E_x=E_x,
            J1=J1,
            J2=J2,
            covHs=covHs,
            covTs=covTs,
        )

        var_deg2 = compute_eq58_half_lambda_torch(
            lam_ext=lam_ext,
            eq52=eq52,
            eq53=eq53,
            theta_deg=doas_deg,
            max_var_deg2=self.max_var_deg2,
        )

        # Final safety clamp.
        # cov_diag_deg2 is the covariance diagonal / variance in deg^2.
        # Max 90 degree uncertainty means max variance = 90^2 = 8100 deg^2.
        #
        # This is already aligned to the external DOA order because Eq.52/Eq.53
        # used lam_ext and V_ext after match_perm_to_external.
        cov_diag_deg2 = var_deg2.to(dtype=doas_deg.dtype).clamp(
            min=0.0,
            max=self.max_var_deg2,
        )
        sigma_deg = torch.sqrt(cov_diag_deg2)


        return sigma_deg, cov_diag_deg2


