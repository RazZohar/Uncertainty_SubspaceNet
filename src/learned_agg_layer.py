import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnedAgg(nn.Module):
    def __init__(self, sensor_count):
        super().__init__()
        self.w = nn.Parameter(torch.empty(0), requires_grad=True)
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
        #if self.w is None or self.d_token != d_token:
        if self.w.numel() == 0:
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


def match_learned_attn_shapes(model, state_dict, prefix="learned_attentaion"):
    for name, param in state_dict.items():
        if name.startswith(prefix) and name.endswith(".w"):
            parts = name.split(".")
            idx = int(parts[1])
            shape = param.shape
            dtype = param.dtype
            device = param.device

            mod = getattr(model, prefix)[idx]

            needs_replacement = (
                not hasattr(mod, "w") or
                not isinstance(mod.w, torch.nn.Parameter) or
                mod.w.shape != shape or
                mod.w.dtype != dtype
            )

            if needs_replacement:
                if dtype.is_complex:
                    # Complex init: match your training init
                    d_token = shape[-1]
                    std = 1.0 / math.sqrt(d_token)
                    real = torch.empty(shape, dtype=torch.float32, device=device).uniform_(-std, std)
                    imag = torch.empty(shape, dtype=torch.float32, device=device).uniform_(-std, std)
                    w = torch.complex(real, imag)
                else:
                    # Real-valued init
                    w = torch.empty(shape, dtype=dtype, device=device)
                    torch.nn.init.xavier_uniform_(w)

                mod.w = torch.nn.Parameter(w)