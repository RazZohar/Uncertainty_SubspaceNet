import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnedAgg(nn.Module):
    def __init__(self, sensor_count):
        super().__init__()
        self.w = None  # lazy init
        self.d_token = None
        self.L = sensor_count

    def forward(self, q: torch.Tensor, rho: torch.Tensor):
        """
        q   : (B, L, H, W)  -- token features
        rho : (B, L, 2)     -- token positions

        Returns
        -------
        z    : (B, D)       -- aggregated representation (over tokens)
        phi  : (B, L)       -- attention weights
        """
        B, L, H, W = q.shape

        # Step 1: flatten q_i
        q_flat = q.view(B, L, -1)        # (B, L, H*W)
        d_q = q_flat.shape[-1]

        # Step 2: concatenate q_i and rho_i
        x = torch.cat([q_flat, rho], dim=-1)  # (B, L, H*W + 2)
        d_token = x.shape[-1]

        # Step 3: lazy init of weight vector
        if self.w is None or self.d_token != d_token:
            std = 1.0 / math.sqrt(d_token)
            real = torch.empty(d_token, dtype=torch.float32, device=x.device).uniform_(-std, std)
            imag = torch.empty(d_token, dtype=torch.float32, device=x.device).uniform_(-std, std)
            w = torch.complex(real, imag)  # dtype: torch.complex64
            self.w = nn.Parameter(w)
            self.register_parameter("w", self.w)
            self.d_token = d_token

        # Step 4: compute logits and softmax
        logits = torch.matmul(x, self.w)      # (B, L)
        phi = F.softmax(logits.real, dim=-1)  # (B, L), real-valued attention weights

        # Weighted sum over q (not x!)
        z = (phi.unsqueeze(-1).unsqueeze(-1) * q).sum(dim=1)  # (B, H, W)

        return z, phi
