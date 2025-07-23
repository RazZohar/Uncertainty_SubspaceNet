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

        if bearings.ndim == 3:
            bearings = bearings.squeeze(-1)  # ensure (B, M)

        # Adjust bearings into [0, pi]
        #bearings = torch.remainder(bearings + torch.pi / 2, torch.pi)

        # Unit direction vectors (B, M, 2)
        d = torch.stack((torch.cos(bearings), torch.sin(bearings)), dim=-1)

        # Rotate 90° to get perpendiculars
        d_perp = torch.stack((-d[..., 1], d[..., 0]), dim=-1)  # (B, M, 2)

        # Normal equations: A @ x = b
        A = d_perp.transpose(1, 2) @ d_perp                    # (B, 2, 2)
        c = (d_perp * positions).sum(dim=2)                    # (B, M)
        b = d_perp.transpose(1, 2) @ c.unsqueeze(-1)           # (B, 2, 1)
        b = b.squeeze(-1)

        I = torch.eye(2, device=positions.device).expand(B, 2, 2)
        A_reg = A + eps * I
        A_inv = torch.linalg.pinv(A_reg)                      # safe inverse
        x_hat = (A_inv @ b.unsqueeze(-1)).squeeze(-1)         # differentiable

        # GDOP: trace(inv(A)), detached
        if self.compute_gdop:
            with torch.no_grad():
                trace = A_inv.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B,)
                gdop = torch.sqrt(trace)
        else:
            gdop = None

        return x_hat, gdop
