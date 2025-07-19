import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAgg(nn.Module):
    """
    phi = softmax( [q_i ; ρ_i] @ W_j ),

    Parameters
    ----------
    d_q   : int   dimensionality of q_i
    d_rho : int   dimensionality of ρ_i
    """
    def __init__(self, d_q: int, d_rho: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_q + d_rho))  # W_j

    def forward(self, q: torch.Tensor, rho: torch.Tensor):
        """
        q   : (batch, L, d_q)
        rho : (batch, L, d_rho)

        Returns
        -------
        z   : (batch, d_q)      weighted sum of q_i
        phi : (batch, L)        softmax weights
        """
        x     = torch.cat([q, rho], dim=-1)          # (B,L,d_q+d_rho)
        logits = torch.matmul(x, self.w)             # (B,L)
        phi    = F.softmax(logits, dim=-1)           # (B,L)
        z      = (phi.unsqueeze(-1) * q).sum(dim=1)  # (B,d_q)
        return z, phi
