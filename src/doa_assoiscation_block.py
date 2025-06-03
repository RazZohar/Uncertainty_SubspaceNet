import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

class SinkhornAssociation(nn.Module):
    """
    Associates unordered DoA estimates from L sub-arrays to a common
    source ordering via a soft permutation (Sinkhorn) layer.

    Parameters
    ----------
    num_src      : int       # J – expected number of sources / bearings per array
    tau          : float     # temperature for Sinkhorn (smaller → harder assignment)
    n_iter       : int       # Sinkhorn normalisation iterations
    anchor_mode  : str       # 'first' | 'centroid'
                              #  'first'    – use sub-array 0 as the reference
                              #  'centroid' – learn a canonical set of angles
    """

    def __init__(self, num_src: int, tau: float = 0.05,
                 n_iter: int = 4, anchor_mode: str = "first"):
        super().__init__()
        self.J        = num_src
        self.tau      = tau
        self.n_iter   = n_iter
        self.anchor   = anchor_mode

        if anchor_mode == "centroid":
            # learnable canonical bearings (initialised uniformly)
            self.register_parameter(
                "phi",
                nn.Parameter(torch.linspace(-pi/2, pi/2, num_src))
            )

    # ---------- helpers -----------------------------------------------------
    @staticmethod
    def _unit(angle):
        """map scalar angle → 2-D unit vector; keeps wrap-arounds smooth"""
        return torch.stack((angle.cos(), angle.sin()), dim=-1)   # (...,2)

    def _sinkhorn(self, cost):
        """
        cost : [L, J, J]  (quadratic distance between local & reference bearings)
        returns doubly-stochastic matrix P  (soft permutation)
        """
        logp = -cost / self.tau
        for _ in range(self.n_iter):
            logp = logp - torch.logsumexp(logp, dim=2, keepdim=True)  # row-norm
            logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)  # col-norm
        return logp.exp()

    # ---------- forward pass ------------------------------------------------
    def forward(self, theta: torch.Tensor,
                sub_pos: torch.Tensor,
                weight: torch.Tensor | None = None):
        """
        theta   : [L, J]          bearings (rad) from each sub-array
        sub_pos : [L, D]          known sensor positions (D = 2 or 3 ≈ unused here)
        weight  : [L, J] | None   optional per-bearing confidence (higher = trust more)

        Returns
        -------
        aligned_theta : [L, J]    bearings reordered to a common source index
        P             : [L, J, J] soft permutation matrices (can be used downstream)
        """
        L, J = theta.shape
        assert J == self.J, "theta second dim ≠ num_src"

        # -- choose reference set φ_j ----------------------------------------
        if self.anchor == "first":
            # sub-array 0 provides the canonical ordering
            ref = theta[0]                         # [J]
        else:  # 'centroid'
            ref = self.phi                        # [J] learnable

        u_ref = self._unit(ref)                   # [J,2]
        u     = self._unit(theta)                 # [L,J,2]

        # -- quadratic distance on the unit circle ---------------------------
        # cost_s[k,j] = ||u_s,k - u_ref_j||^2
        cost = (u.unsqueeze(2) - u_ref)           # [L,J,2] - [J,2] ⇒ [L,J,J,2]
        cost = cost.square().sum(-1)              # [L,J,J]

        # -- optional confidence scaling (smaller cost ↔ higher confidence) --
        if weight is not None:
            # multiply every *row* by w_s,k  ⇒ lower weight = larger cost
            cost = cost * weight.unsqueeze(-1)    # [L,J,1] broadcast

        # -- Sinkhorn soft permutation ---------------------------------------
        P = self._sinkhorn(cost)                  # [L,J,J]

        # -- align bearings ---------------------------------------------------
        aligned = torch.einsum("lkj,lk->lj", P, theta)  # weighted row sum

        return aligned, P
