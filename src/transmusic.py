import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [Seq_len, Batch, d_model]
        return x + self.pe[: x.size(0)]


def calculate_spectrum(En, A_grid):
    """
    Calculates the MUSIC pseudo-spectrum dynamically.
    En: [Batch, M, M]
    A_grid: [Batch, M, Num_Angles]
    """
    En_H = torch.conj(En.transpose(1, 2))
    H1 = torch.bmm(torch.bmm(En, En_H), A_grid)
    H2 = H1 * torch.conj(A_grid)
    H3 = torch.sum(H2, dim=1)
    return 1.0 / torch.clamp(torch.abs(H3), min=1e-12)


class TransMUSIC(nn.Module):
    """
    TransMUSIC benchmark model with dual heads for DoA and uncertainty.

    Input samples may be [B, S, R, N, T], [B, S, N, T], or [B, N, T].
    Output is always trainer-compatible: bearings and sigma_i shaped [B, S, M].
    """

    def __init__(self, num_antennas, num_sources, num_angle_bins=360, d_spacing=0.5, nhead=8):
        super(TransMUSIC, self).__init__()

        self.num_antennas = int(num_antennas)
        self.num_sources = int(num_sources)
        self.num_angle_bins = int(num_angle_bins)
        self.d_spacing = float(d_spacing)

        self.d_model = 2 * self.num_antennas
        if self.d_model % nhead != 0:
            # Keep Transformer valid for non-8-antenna experiments.
            valid_heads = [h for h in (8, 4, 2, 1) if self.d_model % h == 0]
            nhead = valid_heads[0]
        self.nhead = nhead

        self.BN = nn.BatchNorm1d(self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=1024,
            dropout=0.0,
            activation="relu",
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc_vector_dim = 2 * self.num_antennas * self.num_antennas
        self.input_linear = nn.Linear(in_features=self.d_model, out_features=self.fc_vector_dim)

        self.doa_head = nn.Sequential(
            nn.Linear(in_features=self.num_angle_bins, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=self.num_sources),
        )

        self.uncertainty_head = nn.Sequential(
            nn.Linear(in_features=self.num_angle_bins, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=self.num_sources),
            nn.Softplus(),
        )

    @staticmethod
    def _prepare_samples(samples: torch.Tensor) -> torch.Tensor:
        """Return complex IQ samples as [B, S, N, T]."""
        if samples.dim() == 5:
            return samples[:, :, 0, :, :]
        if samples.dim() == 4:
            return samples
        if samples.dim() == 3:
            return samples.unsqueeze(1)
        raise ValueError(f"Unsupported samples shape for TransMUSIC: {tuple(samples.shape)}")

    def _generate_steering_grid(self, batch_size, device):
        """Generates the ULA steering matrix grid for the pseudo-spectrum."""
        angles = torch.linspace(-torch.pi / 2, torch.pi / 2, self.num_angle_bins, device=device)
        m = torch.arange(self.num_antennas, device=device).float()
        phases = -1j * 2 * torch.pi * self.d_spacing * torch.outer(m, torch.sin(angles))
        A_grid = torch.exp(phases)
        return A_grid.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, sensor_positions, samples, doa_gt=None):
        x = self._prepare_samples(samples)
        B, S, N, T = x.shape
        if N != self.num_antennas:
            raise ValueError(f"TransMUSIC expected N={self.num_antennas} antennas, got N={N}.")

        device = samples.device
        x = x.reshape(B * S, N, T)

        # Concat real and imaginary parts: [B*S, 2N, T]
        x = torch.cat([x.real, x.imag], dim=1).float()
        x = self.BN(x)

        # Transformer expects [T, B*S, 2N]
        x = x.permute(2, 0, 1)
        x = self.pos_encoder(x)
        x_encoded = self.encoder(x)

        # Mean-pool snapshots -> [B*S, 2N]
        x_pooled = torch.mean(x_encoded, dim=0)

        # Pseudo-noise-subspace components -> [B*S, N, N]
        feature_vector = self.input_linear(x_pooled)
        x_reshaped = feature_vector.reshape(B * S, 2 * N, N)
        En = torch.complex(x_reshaped[:, :N, :], x_reshaped[:, N:, :])

        A_grid = self._generate_steering_grid(B * S, device)
        spectrum = calculate_spectrum(En, A_grid)

        doa_preds = self.doa_head(spectrum).reshape(B, S, self.num_sources)
        sigma_deg = (self.uncertainty_head(spectrum) + 1e-4).reshape(B, S, self.num_sources)

        return {
            "bearings": doa_preds,
            "sigma_i": sigma_deg,
        }
