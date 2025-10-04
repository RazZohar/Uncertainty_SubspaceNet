import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

class FixedVectorQuantizer(nn.Module):
    """
    Fixed vector quantizer: Get Fixed Rate Quantizer
    """
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33):
        super(FixedVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # Initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)

        # Balancing parameter lambda for the commitment loss
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

        # Track codebook usage
        self.register_buffer("codebook_usage", torch.zeros(codebook_size))
        self.general_codebook_usage = 0

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
            torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.d)

        # Use the entire codebook for quantization
        actives = self.codebook.weight

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(actives ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, actives.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.p, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

        # Track codebook usage
        self.update_codebook_usage(encoding_indices)
        self.general_codebook_usage = torch.unique(encoding_indices).numel()


        if self.training:
            # Loss
            q_latent_loss = torch.nn.functional.mse_loss(quantized, inputs.detach())  # Commitment loss
            e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)  # Alignment loss
            cb_loss = q_latent_loss + self.lambda_c * e_latent_loss  # Codebook loss

            # Gradient copying for the straight-through estimator
            quantized = inputs + (quantized - inputs).detach()
        else:
            cb_loss = 0

        """
        # Do the rotation trick from  RESTRUCTURING VECTOR QUANTIZATION WITH THE
        # ROTATION TRICK
        # https://arxiv.org/pdf/2410.06424
        pre_norm_q = self.get_very_efficient_rotation(inputs / (torch.norm(inputs, dim=1, keepdim=True) + 1e-6),
                                                      quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                                                      inputs.unsqueeze(1)).squeeze()
        quantized = pre_norm_q * (
                torch.norm(quantized, dim=1, keepdim=True) / (torch.norm(inputs, dim=1, keepdim=True) + 1e-6)).detach()
        """
        return quantized, cb_loss

    # By product of this function is uninatalized codebook
    def set_codebook_size(self, codebook_size):
        self.p = codebook_size

        # Initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)

        # Track codebook usage
        self.register_buffer("codebook_usage", torch.zeros(codebook_size))

    def update_codebook_usage(self, indices):
        # Count usage of each codebook entry
        with torch.no_grad():
            flattened_indices = indices.view(-1)
            counts = torch.bincount(flattened_indices, minlength=self.p)
            self.codebook_usage += counts

    def reset_codebook_usage(self):
        # Reset codebook usage tracking
        self.codebook_usage.zero_()

    def visualize_codebook_usage(self):
        # Visualize codebook usage
        return self.codebook_usage.cpu().numpy()



# This is the element wise qunatizer
class ElementWiseQuantizer(nn.Module):
    def __init__(self, n_levels=16, min_val=-1.0, max_val=1.0):
        """
        Element-wise uniform quantizer for complex numbers.
        :param n_levels: Number of quantization levels (bins)
        :param min_val: Minimum value to be quantized (applies to real and imaginary separately)
        :param max_val: Maximum value to be quantized (applies to real and imaginary separately)
        """
        super().__init__()
        self.n_levels = n_levels
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = (max_val - min_val) / (n_levels - 1)  # Step size

        # Tracking histograms for real and imaginary parts
        self.register_buffer("quantization_counts_real", torch.zeros(n_levels))
        self.register_buffer("quantization_counts_imag", torch.zeros(n_levels))


    def init_limits(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = (max_val - min_val) / (self.n_levels - 1)  # Step size
        

    def quantize(self, x, counts):
        """
        Helper function to quantize real or imaginary part separately.
        :param x: Input tensor (real or imaginary part)
        :param counts: Tracking tensor for quantization usage
        :return: Quantized tensor
        """
        # Adaptive Uniform quantization
        #self.init_limits(x)
        # Use Noise with the quantization distrbuited uniformally
        x = x #+ ((self.max_val - self.min_val) * torch.rand(x.shape) + self.min_val)

        with torch.no_grad():
            x_clamped = torch.clamp(x, self.min_val, self.max_val)  # Clip values
            x_normalized = (x_clamped - self.min_val) / self.step_size  # Normalize to [0, n_levels-1]
            x_rounded = torch.round(x_normalized)  # Round to nearest quantization bin
            x_quantized = x_rounded * self.step_size + self.min_val  # Convert back to real value

        # Track histogram of quantized values
        """with torch.no_grad():  # No gradients needed for monitoring
            indices = x_rounded.long().flatten()  # Convert to integer indices
            valid_mask = (indices >= 0) & (indices < self.n_levels)  # Ensure valid indices
            counts.scatter_add_(0, indices[valid_mask], torch.ones_like(indices[valid_mask]))
        """
        #print(f'original {x=} when {x_quantized=} diff={x-x_quantized}')
        return x_quantized

    def forward(self, x: torch.Tensor):
        """
        Quantizes a complex input tensor element-wise.
        :param x: Complex tensor of shape (batch, features), where features contain complex values.
        :return: Quantized complex tensor
        """
        real_part = self.quantize(x.real, self.quantization_counts_real)
        #imag_part = self.quantize(x.imag, self.quantization_counts_imag)
        #return torch.complex(real_part, imag_part), 0  # Reconstruct quantized complex tensor
        return real_part, 0

    def get_usage_distribution(self):
        """
        Returns the current distribution of quantization level usage for real and imaginary parts.
        """
        real_dist = self.quantization_counts_real / self.quantization_counts_real.sum()
        imag_dist = self.quantization_counts_imag / self.quantization_counts_imag.sum()
        return real_dist, imag_dist

    def plot_usage_distribution(self):
        """
        Plots the histogram of quantization level usage for both real and imaginary parts.
        """
        real_dist, imag_dist = self.get_usage_distribution()
        levels = torch.linspace(self.min_val, self.max_val, self.n_levels).cpu().numpy()

        plt.figure(figsize=(10, 5))

        # Real Part Usage
        plt.subplot(1, 2, 1)
        plt.bar(levels, real_dist.cpu().numpy(), width=self.step_size * 0.8, color="b", alpha=0.7)
        plt.xlabel("Real Part Quantization Levels")
        plt.ylabel("Usage Frequency")
        plt.title("Real Part Quantization Level Usage")
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Imaginary Part Usage
        plt.subplot(1, 2, 2)
        plt.bar(levels, imag_dist.cpu().numpy(), width=self.step_size * 0.8, color="r", alpha=0.7)
        plt.xlabel("Imaginary Part Quantization Levels")
        plt.ylabel("Usage Frequency")
        plt.title("Imaginary Part Quantization Level Usage")
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.show()



class AdaptiveVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, codebook_size, lambda_c=0.1, lambda_p=0.33):
        super(AdaptiveVectorQuantizer, self).__init__()

        self.d = num_embeddings  # The size of the vectors
        self.p = codebook_size  # Number of vectors in the codebook

        # initialize the codebook
        self.codebook = nn.Embedding(self.p, self.d)
        self.codebook.weight.data.uniform_(-1 / self.p, 1 / self.p)

        # Balancing parameter lambda for the commintment loss
        self.lambda_c = lambda_c
        self.lambda_p = lambda_p

        self.first = True

    def forward(self, inputs, num_vectors, prev_vecs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self.d)

        quant_vecs = []
        losses = []

        for num_actives in range(int(np.log2(num_vectors))):
            actives = self.codebook.weight[:pow(2, num_actives + 1)]

            # Calculate distances
            distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                         + torch.sum(actives ** 2, dim=1)
                         - 2 * torch.matmul(flat_input, actives.t()))

            # Encoding
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(encoding_indices.shape[0], self.p, device=inputs.device)
            encodings.scatter_(1, encoding_indices, 1)

            # Quantize and unflatten
            quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)
            quant_vecs.append(quantized)

        for num_actives in range(int(np.log2(num_vectors))):
            if self.training:
                # Loss
                q_latent_loss = F.mse_loss(quant_vecs[num_actives], inputs.detach())  # commitment loss

                if num_actives == 0:
                    prox_loss = 0
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)  # alignment loss

                elif num_actives == 1:
                    e_latent_loss = F.mse_loss(quant_vecs[num_actives].detach(), inputs)
                    prox_loss = (num_actives * self.lambda_p) * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                                           actives[:pow(2, num_actives + 1) // 2])
                else:
                    e_latent_loss = 0
                    prox_loss = self.lambda_p * F.mse_loss(prev_vecs[:pow(2, num_actives + 1) // 2],
                                                           actives[:pow(2, num_actives + 1) // 2])  # proximity_loss

                cb_loss = q_latent_loss + self.lambda_c * e_latent_loss + prox_loss  # codebook loss

                quant_vecs[num_actives] = inputs + (quant_vecs[num_actives] - inputs).detach()  # gradient copying

            else:
                cb_loss = 0

            losses.append(cb_loss)

        return quant_vecs, losses, actives
