import torch
import torch.nn as nn

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
