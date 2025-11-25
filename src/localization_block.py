import torch
import torch.nn as nn
import math

class RayIntersection(nn.Module):
    """
    Batched least-squares intersection of 2D rays.
    No learnable parameters. Fully differentiable output (x_hat).
    GDOP is detached from autograd for safe logging.

    Inputs:
    -------
    positions : (B, M, 2) - sensor positions
    bearings  : (B, M) or (B, M, 1) - angles [rad]

    Returns:
    --------
    x_hat : (B, 2) - estimated intersection point
    gdop  : (B,)   - geometric dilution of precision (no gradients)
    """
    def __init__(self, compute_gdop: bool = True, eps: float = 1e-6):
        super().__init__()
        self.compute_gdop = compute_gdop
        self.eps = eps

    def forward(self, positions: torch.Tensor, bearings: torch.Tensor):
        B, M, _ = positions.shape
        eps = self.eps

        # ----- bearings shape handling -----
        # (B, M)  -> (B, M, 1)
        # (B, M, K) stays as-is
        if bearings.ndim == 2:
            bearings = bearings.unsqueeze(-1)  # (B, M, 1)
        elif bearings.ndim != 3:
            raise ValueError(f"bearings must be (B,M) or (B,M,K), got {bearings.shape}")

        _, _, K = bearings.shape

        # ----- unit direction & perpendicular vectors -----
        # d: (B, M, K, 2)
        d = torch.stack((torch.cos(bearings), torch.sin(bearings)), dim=-1)

        # d_perp: (B, M, K, 2)
        d_perp = torch.stack((-d[..., 1], d[..., 0]), dim=-1)

        # Reorder to work per-mode:
        # dpk: (B, K, M, 2)
        dpk = d_perp.permute(0, 2, 1, 3)

        # positions: (B, M, 2) -> (B, 1, M, 2) broadcast over K
        p = positions.unsqueeze(1)  # (B, 1, M, 2) -> (B, K, M, 2)

        # ----- Normal equations: A x = b  (per batch & per mode) -----
        # A = Σ_m n_m n_m^T
        # dpk^T: (B, K, 2, M), dpk: (B, K, M, 2) => A: (B, K, 2, 2)
        A = dpk.transpose(-1, -2) @ dpk  # (B, K, 2, 2)

        # q_m = n_m^T p_m  (scalar per m)
        # (B, K, M, 2) * (B, K, M, 2) -> (B, K, M)
        q = (dpk * p).sum(dim=-1)  # (B, K, M)

        # b = Σ_m n_m q_m = N^T q
        # dpk^T: (B, K, 2, M), q: (B, K, M, 1)
        b = (dpk.transpose(-1, -2) @ q.unsqueeze(-1)).squeeze(-1)  # (B, K, 2)

        # ----- Regularization & solve -----
        I = torch.eye(2, device=positions.device).view(1, 1, 2, 2)  # (1, 1, 2, 2)
        A_reg = A + eps * I  # (B, K, 2, 2)

        A_inv = torch.linalg.pinv(A_reg)  # (B, K, 2, 2)
        x_hat = (A_inv @ b.unsqueeze(-1)).squeeze(-1)  # (B, K, 2)

        # If only one mode, squash K dimension to keep old API: (B, 2)
        if K == 1:
            x_hat = x_hat.squeeze(1)  # (B, 2)

        # ----- GDOP: sqrt(trace(inv(A))) -----
        if self.compute_gdop:
            with torch.no_grad():
                trace = A_inv.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, K)
                gdop = torch.sqrt(trace)  # (B, K)
                if K == 1:
                    gdop = gdop.squeeze(1)  # (B,)
        else:
            gdop = None

        return x_hat, gdop



import torch

def triangulation_with_soft_area_batched(
    sensor_positions,   # (B, 2, 2)
    bearings,           # (B, 2, T)
    sigmas,             # (B, 2, T)
    eps=1e-8,
):
    """
    Differentiable triangulation + 'soft area' for batched bearings
    with *batched* sensor positions.

    Parameters
    ----------
    sensor_positions : (B, 2, 2)
        Per-batch sensor positions: [ [x1,y1], [x2,y2] ] for each batch.
    bearings : (B, 2, T)
        Bearings (radians) from each sensor to each target.
    sigmas : (B, 2, T)
        Bearing std-dev (radians), same shape as bearings.
    eps : float
        Numerical stabilizer.

    Returns
    -------
    centroid : (B, T, 2)
        Estimated source position per (batch, target).
    area_soft : (B, T)
        Soft 'intersection area' ≈ π * sqrt(det(Σ_s)).
    Sigma_s : (B, T, 2, 2)
        Covariance of source location.
    """
    if bearings.shape != sigmas.shape:
        raise ValueError("bearings and sigmas must have the same shape (B, 2, T).")
    if bearings.shape[1] != 2:
        raise ValueError("This implementation assumes exactly 2 sensors (sensor_count=2).")
    if sensor_positions.ndim != 3 or sensor_positions.shape[1:] != (2, 2):
        raise ValueError("sensor_positions must have shape (B, 2, 2).")

    B, S, T = bearings.shape  # S == 2

    # Sensor positions per batch
    c1 = sensor_positions[:, 0, :]          # (B, 2)
    c2 = sensor_positions[:, 1, :]          # (B, 2)

    B_vec = c2 - c1                         # (B, 2)
    Bx = B_vec[..., 0].unsqueeze(-1)        # (B, 1)
    By = B_vec[..., 1].unsqueeze(-1)        # (B, 1)

    # Bearings and sigmas
    beta1 = bearings[:, 0, :]               # (B, T)
    beta2 = bearings[:, 1, :]               # (B, T)

    sigma1 = sigmas[:, 0, :]                # (B, T)
    sigma2 = sigmas[:, 1, :]                # (B, T)

    # ---- 1) Triangulation ----
    c1b = torch.cos(beta1)                  # (B, T)
    s1b = torch.sin(beta1)
    c2b = torch.cos(beta2)
    s2b = torch.sin(beta2)

    d1 = torch.stack((c1b, s1b), dim=-1)    # (B, T, 2)
    d2 = torch.stack((c2b, s2b), dim=-1)    # (B, T, 2)

    D = torch.sin(beta2 - beta1)            # (B, T)
    D_safe = D + eps * torch.sign(D + 1e-12)

    # N = cross(B, d2) with B per batch
    N = Bx * s2b - By * c2b                 # (B, T)

    t1 = N / D_safe                         # (B, T)

    # Broadcast c1 to (B, T, 2)
    c1_bt = c1.unsqueeze(1)                 # (B, 1, 2) -> broadcast with (B, T, 2)
    s = c1_bt + t1.unsqueeze(-1) * d1       # (B, T, 2)

    # ---- 2) Jacobian J = ∂s / ∂(beta1, beta2) ----
    cos_d = torch.cos(beta2 - beta1)        # (B, T)

    dN_db1 = torch.zeros_like(beta1)        # 0
    dN_db2 = Bx * c2b + By * s2b            # (B, T)

    dD_db1 = -cos_d                         # (B, T)
    dD_db2 =  cos_d                         # (B, T)

    dt1_db1 = (dN_db1 * D_safe - N * dD_db1) / (D_safe ** 2)  # (B, T)
    dt1_db2 = (dN_db2 * D_safe - N * dD_db2) / (D_safe ** 2)  # (B, T)

    # ∂d1/∂beta1 = (-sinβ1, cosβ1)
    dd1_db1 = torch.stack((-s1b, c1b), dim=-1)                # (B, T, 2)

    ds_db1 = dt1_db1.unsqueeze(-1) * d1 + t1.unsqueeze(-1) * dd1_db1  # (B, T, 2)
    ds_db2 = dt1_db2.unsqueeze(-1) * d1                                  # (B, T, 2)

    J = torch.stack((ds_db1, ds_db2), dim=-1)              # (B, T, 2, 2)

    # ---- 3) Bearing covariance -> source covariance ----
    var1 = sigma1 ** 2                                     # (B, T)
    var2 = sigma2 ** 2                                     # (B, T)

    Sigma_beta = torch.zeros(B, T, 2, 2,
                             device=bearings.device,
                             dtype=bearings.dtype)
    Sigma_beta[..., 0, 0] = var1
    Sigma_beta[..., 1, 1] = var2                          # (B, T, 2, 2)

    Sigma_s = J @ Sigma_beta @ J.transpose(-1, -2)        # (B, T, 2, 2)

    # ---- 4) Soft area = π * sqrt(det(Sigma_s)) ----
    a = Sigma_s[..., 0, 0]
    b = Sigma_s[..., 0, 1]
    c = Sigma_s[..., 1, 0]
    d = Sigma_s[..., 1, 1]

    det_S = a * d - b * c
    det_S = det_S.clamp_min(eps)                          # (B, T)

    area_soft = torch.pi * torch.sqrt(det_S)              # (B, T)

    return s, area_soft, Sigma_s
