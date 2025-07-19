"""Subspace-Net 
Details
----------
Name: models.py
Authors: New by Raz Zohar
Created: 01/10/21
Edited: 22/06/2025

Purpose:
--------
This script defines the tested NN-models and the model-based DL models, which used for simulation.
The implemented models:
    * DeepRootMUSIC: model-based deep learning algorithm as described in:
        [1] D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun and N. Shlezinger,
        "Deep Root Music Algorithm for Data-Driven Doa Estimation," ICASSP 2023 - 
        2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
        Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096504.
        
    * SubspaceNet: model-based deep learning algorithm as described in:
        [2] "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
    
    * DA-MUSIC: Deep Augmented MUSIC model-based deep learning algorithm as described in
        [3] J. P. Merkofer, G. Revach, N. Shlezinger, and R. J. van Sloun, “Deep
        augmented MUSIC algorithm for data-driven DoA estimation,” in IEEE
        International Conference on Acoustics, Speech and Signal Processing
        (ICASSP), 2022, pp. 3598-3602."
        
    * DeepCNN: Deep learning algorithm as described in:
        [4] G. K. Papageorgiou, M. Sellathurai, and Y. C. Eldar, “Deep networks
        for direction-of-arrival estimation in low SNR,” IEEE Trans. Signal
        Process., vol. 69, pp. 3714-3729, 2021.

Functions:
----------
This script also includes the implementation of Root-MUSIC algorithm, as it is written using Pytorch library,
for the usage of src.models: SubspaceNet implementation.
"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import warnings
import math

from torch.ao.quantization import quantize

import src.qunatizer
from src.utils import gram_diagonal_overload, device, add_epsilon_batch
from src.utils import sum_of_diags_torch, find_roots_torch

from src.qunatizer import FixedVectorQuantizer, AdaptiveVectorQuantizer
from src.data_handler import create_autocorrelation_tensor
#from src.task_ignorant_model import SignalsSubspaceNetEsprit

warnings.simplefilter("ignore")
# Constants
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelGenerator(object):
    """
    Generates an instance of the desired model, according to model configuration parameters.
    """

    def __init__(self):
        """
        Initialize ModelParams object.
        """
        self.model_type = None
        self.diff_method = None
        self.tau = None

    def set_tau(self, tau: int = None):
        """
        Set the value of tau parameter for SubspaceNet model.

        Parameters:
            tau (int): The number of lags.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If tau parameter is not provided for SubspaceNet model.
        """
        if self.model_type.startswith("SubspaceNet"):
            if not isinstance(tau, int):
                raise ValueError(
                    "ModelParams.set_tau: tau parameter must be provided for SubspaceNet model"
                )
            self.tau = tau
        return self

    def set_diff_method(self, diff_method: str = "root_music"):
        """
        Set the differentiation method for SubspaceNet model.

        Parameters:
            diff_method (str): The differantiable subspace method ("esprit" or "root_music").

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If the diff_method is not defined for SubspaceNet model.
        """
        if self.model_type.startswith("SubspaceNet"):
            if diff_method not in ["esprit", "root_music"]:
                raise ValueError(
                    f"ModelParams.set_diff_method: {diff_method} is not defined for SubspaceNet model"
                )
            self.diff_method = diff_method
        return self

    def set_model_type(self, model_type: str):
        """
        Set the model type.

        Parameters:
            model_type (str): The model type.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If model type is not provided.
        """
        if not isinstance(model_type, str):
            raise ValueError(
                "ModelParams.set_model_type: model type has not been provided"
            )
        self.model_type = model_type
        return self

    def set_model(self, system_model_params):
        """
        Set the model based on the model type and system model parameters.

        Parameters:
            system_model_params (SystemModelParams): The system model parameters.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            Exception: If the model type is not defined.
        """
        if self.model_type.startswith("DA-MUSIC"):
            self.model = DeepAugmentedMUSIC(
                N=system_model_params.N,
                T=system_model_params.T,
                M=system_model_params.M,
            )
        elif self.model_type.startswith("DeepCNN"):
            self.model = DeepCNN(N=system_model_params.N, grid_size=361)
        elif self.model_type.startswith("SubspaceNet"):
            self.model = SubspaceNet(
                tau=self.tau, M=system_model_params.M, diff_method=self.diff_method,
                quantize=False, codebook_size=system_model_params.codebook_size
            )
        elif self.model_type.startswith("SignalsSubspaceNet"):
            self.model = SignalsSubspaceNetEsprit(N=system_model_params.N, T=system_model_params.T, tau=self.tau, M=system_model_params.M, codebook_size=system_model_params.codebook_size)
            #self.model.set_diff_method(diff_method=self.diff_method)
        elif self.model_type.startswith("TaskIgnorantSubspaceNet"):
            self.model = TaskIgnorantSubspaceNet(N=system_model_params.N, T=system_model_params.T, tau=self.tau, M=system_model_params.M, codebook_size=system_model_params.codebook_size)
            #self.model.set_diff_method(diff_method=self.diff_method)
        else:
            raise Exception(
                f"ModelGenerator.set_model: Model type {self.model_type} is not defined"
            )
        return self


class DeepRootMUSIC(nn.Module):
    """DeepRootMUSIC is model-based deep learning model for DOA estimation problem.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        LeakyReLU (nn.LeakyReLU): Leaky reLu activation function, with activation_value.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, activation_value: float):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            activation_value (float): Value for the activation function.

        """
        super(DeepRootMUSIC, self).__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=2)
        self.LeakyReLU = nn.LeakyReLU(activation_value)
        self.DropOut = nn.Dropout(0.2)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the DeepRootMUSIC.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.LeakyReLU(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        # DCNN block #1
        x = self.deconv1(x)
        x = self.LeakyReLU(x)
        # DCNN block #2
        x = self.deconv2(x)
        x = self.LeakyReLU(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv3(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx_tag, eps=1)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_prediction, doa_all_predictions, roots = root_music(
            Rz, self.M, self.batch_size
        )
        return doa_prediction, doa_all_predictions, roots, Rz

class FixedVectorQuantizer(nn.Module):
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
        # Ensure e is (N, 1, C)
        if e.dim() == 2:
            e = e.unsqueeze(1)  # Convert (N, C) → (N, 1, C)

        # Ensure u and q are (N, C, 1) for correct bmm usage
        u = u.unsqueeze(-1).detach()  # (N, C, 1)
        q = q.unsqueeze(1).detach()  # (N, 1, C)
        w = w.unsqueeze(-1)  # (N, C, 1)

        # First reflection term
        e = e - 2 * torch.bmm(torch.bmm(e, w), w.transpose(1, 2))

        # Second rotation term
        e = e + 2 * torch.bmm(torch.bmm(e, u), q)

        return e.squeeze(1)  # Convert back to (N, C) if necessary


    def apply_rotation_trick(self, inputs, quantized):
        """
        Notes: The rotation trick is implmented here but for my setup the results are worsen
        # Do the rotation trick from  RESTRUCTURING VECTOR QUANTIZATION WITH THE
        # ROTATION TRICK
        # https://arxiv.org/pdf/2410.06424
        """
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        b, c, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(-1, c)  # (b, h, w, c) -> (b*h*w, c)
        quantized = quantized.permute(0, 2, 3, 1).reshape(-1, c)  # (b, h, w, c) -> (b*h*w, c)

        pre_norm_q = self.get_very_efficient_rotation(inputs / (torch.norm(inputs, dim=1, keepdim=True) + 1e-6),
                                                     quantized / (torch.norm(quantized, dim=1, keepdim=True) + 1e-6),
                                                     inputs.unsqueeze(1)).squeeze()
        quantized = pre_norm_q * (
               torch.norm(quantized, dim=1, keepdim=True) / (torch.norm(inputs, dim=1, keepdim=True) + 1e-6)).detach()


        quantized = quantized.view(b, h, w, c).permute(0, 2, 3, 1)
        return quantized



    def forward(self, inputs):
        input_shape = inputs.shape
        #inputs = inputs.permute(0, 2, 3, 1).contiguous()

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


class AntiRectifierLayer(nn.Module):
    def __init__(self, function):
        super(AntiRectifierLayer, self).__init__()
        self.function = function
    def forward(self, x):
        return self.function(x)



class ComplexReLU(nn.Module):
    def __init__(self, function):
        super(ComplexReLU, self).__init__()
        self.function = function
    def forward(self, X):
        return torch.complex(self.function(X.real), self.function(X.imag))



# TODO: inherit SubspaceNet from DeepRootMUSIC
class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, M: int, diff_method: str = "root_music", quantize: bool = False, codebook_size: int = 256):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__()
        self.M = M
        self.tau = tau
        self.quantize = quantize
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)

        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.anti_rectifier_layer = AntiRectifierLayer(self.anti_rectifier)

        self.encoder = nn.Sequential(self.conv1,
                                     #self.batchnorm1,
                                     self.anti_rectifier_layer,
                                     self.conv2,
                                     #self.batchnorm2,
                                     self.anti_rectifier_layer,
                                     self.conv3)

        num_embeddings = 4
        self.codebook_size = codebook_size
        lambda_c = 0.1
        lambda_p = 0.33
        self.quantizer = FixedVectorQuantizer(num_embeddings, self.codebook_size, lambda_c, lambda_p)

        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()

        # Define The decoder of the AE architecture
        self.decoder = nn.Sequential(self.anti_rectifier_layer,
                                     self.deconv2,
                                     self.anti_rectifier_layer,
                                     self.deconv3,
                                     self.anti_rectifier_layer,
                                     self.DropOut,
                                     self.deconv4)

        self.__unique_indices_set = set()

        # Set the subspace method for training
        self.set_diff_method(diff_method)

    def set_diff_method(self, diff_method: str):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if diff_method.startswith("root_music"):
            self.diff_method = root_music
        elif diff_method.startswith("esprit"):
            self.diff_method = esprit
        else:
            raise Exception(
                f"SubspaceNet.set_diff_method: Method {diff_method} is not defined for SubspaceNet"
            )

    def set_quantize(self, quantize: bool):
        self.quantize = quantize

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]

        ## Architecture flow ##
        # Apply The encoder from the AE architecture
        x = self.encoder(Rx_tau)

        x_normalized = x - x.mean()

        # quantize if needed
        if self.quantize:
            z_quantized, vq_loss = self.quantizer(x_normalized)

            self.__unique_indices_set.update(torch.unique(z_quantized).tolist())
            self.codebook_utilization = len(self.__unique_indices_set) / self.codebook_size
        else:
            z_quantized, vq_loss = x_normalized, 0

        # Apply the decoder from the AE architecture
        Rx = self.decoder(z_quantized)

        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to the differentiable subspace algorithm
        method_output = self.diff_method(Rz, self.M, self.batch_size)
        if isinstance(method_output, tuple) and len(method_output[0]) > 1:
            # Root MUSIC output
            doas, subspace_info = method_output
            doa_prediction, doa_all_predictions, roots = doas
        else:
            # Esprit output
            doa_prediction, subspace_information = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz, vq_loss


class SubspaceNetEsprit(SubspaceNet):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.
        SubspaceNetEsprit is based on the ability to perform back-propagation using ESPRIT algorithm,
        instead of RootMUSIC.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.

    """

    def __init__(self, tau: int, M: int):
        super().__init__(tau, M)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]

        ## Architecture flow ##
        # Apply The encoder from the AE architecture
        x = self.encoder(Rx_tau)

        # Quantize
        x_normalized = x - x.mean()

        # quantize if needed
        if self.quantize:
            z_quantized, vq_loss = self.quantizer(x_normalized)

            self.__unique_indices_set.update(torch.unique(z_quantized).tolist())
            self.codebook_utilization = len(self.__unique_indices_set) / self.codebook_size
        else:
            z_quantized, vq_loss = x_normalized, 0

        # Apply the decoder from the AE architecture
        Rx = self.decoder(z_quantized)

        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Esprit algorithm
        doa_prediction, estimated_subspace = esprit(Rz, self.M, self.batch_size)
        return doa_prediction, Rz, vq_loss

class ComplexConv1d(nn.Module):
    """ Complex-valued 1D Convolution Layer """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.activation = nn.ReLU()

    def forward(self, X):
        real = self.conv_real(X.real) - self.conv_imag(X.imag)
        imag = self.conv_real(X.imag) + self.conv_imag(X.real)
        return torch.complex(real, imag)

class ComplexConvTranspose1d(nn.Module):
    """ Complex-valued 1D Transposed Convolution Layer """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.deconv_real = nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
        self.deconv_imag = nn.ConvTranspose1d(in_channels, out_channels, kernel_size)
        self.activation = nn.ReLU()

    def forward(self, X):
        real = self.deconv_real(X.real) - self.deconv_imag(X.imag)
        imag = self.deconv_real(X.imag) + self.deconv_imag(X.real)
        return torch.complex(real, imag)




class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, X):
        return torch.complex(self.dropout(X.real), self.dropout(X.imag))

class SignalsSubspaceNetEsprit(SubspaceNetEsprit):
    """
    SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.
        SubspaceNetEsprit is based on the ability to perform back-propagation using ESPRIT algorithm,
        instead of RootMUSIC.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.

    """

    def __init__(self, N: int, T: int, tau: int, M: int, quantize_source=False, codebook_size=256):
        super().__init__(8, M)
        self.quantize_source = quantize_source

        self.N = N
        self.T = T

        in_channels = 8
        hidden_channels = 32
        out_channels = 8
        latent_dim=16

        # TODO: check sizes of Conv


        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.complex_rectifier = ComplexReLU(self.anti_rectifier)

        self.anti_rectifier_layer = AntiRectifierLayer(self.complex_rectifier)

        self.conv1 = ComplexConv1d(in_channels, 16, kernel_size=2)
        self.conv2 = ComplexConv1d(32, 32, kernel_size=2)
        self.conv3 = ComplexConv1d(64, 64, kernel_size=2)

        self.deconv2 = ComplexConvTranspose1d(128, 32, kernel_size=2)
        self.deconv3 = ComplexConvTranspose1d(64, 16, kernel_size=2)
        self.deconv4 = ComplexConvTranspose1d(32, out_channels, kernel_size=2)

        self.encoder_signal = nn.Sequential(self.conv1,
                                            # self.batchnorm1,
                                            self.anti_rectifier_layer,
                                            self.conv2,
                                            # self.batchnorm2,
                                            self.anti_rectifier_layer,
                                            self.conv3)


        self.DropOut = ComplexDropout(0.2)
        self.ReLU = nn.ReLU()

        # Define The decoder of the AE architecture
        self.decoder_signal = nn.Sequential(self.anti_rectifier_layer,
                                            self.deconv2,
                                            self.anti_rectifier_layer,
                                            self.deconv3,
                                            self.anti_rectifier_layer,
                                            self.DropOut,
                                            self.deconv4)

        num_embeddings = 4
        self.codebook_size = codebook_size
        lambda_c = 0.1
        lambda_p = 0.33
        self.quantizer_signal = FixedVectorQuantizer(num_embeddings, self.codebook_size, lambda_c, lambda_p)

        self.__unique_indices_set = set()
        self.online_inference = False

    def set_quantize(self, quantize: bool):
        self.quantize_source = quantize

    def forward(self, x: torch.Tensor):


        # part of the encoder is used as the sensed device
        vq_loss, z_quantized = self.sense_device_forward(x)

        # The inference device recieve the quantized word and then decode and apply subspace
        Rz, doa_prediction = self.inference_device_forward(z_quantized)

        return doa_prediction, Rz, vq_loss

    def inference_device_forward(self, z_quantized):
        x_hat = self.decoder_signal(z_quantized)
        # Create the auto-correlation matrix out of after the quantization
        Rx_matrix = self.calculate_cov_batch(x_hat)
        # Update progressivly if enabled
        Rx_matrix = self.calculate_progressive_coveriance(Rx_matrix)
        # Apply Gram operation diagonal loading
        # Rz = gram_diagonal_overload(
        #    Kx=Rx_matrix, eps=1, batch_size=self.batch_size
        # )
        Rz = add_epsilon_batch(Kx=Rx_matrix, eps=1, batch_size=self.batch_size)
        # Feed surrogate covariance to the differentiable subspace algorithm
        """method_output = self.diff_method(Rz, self.M, self.batch_size)
                if isinstance(method_output, tuple):
                    # Root MUSIC output
                    doa_prediction, doa_all_predictions, roots = method_output
                else:
                    # Esprit output
                    doa_prediction = method_output
                    doa_all_predictions, roots = None, None
                return doa_prediction, doa_all_predictions, roots, Rz, vq_loss
                """
        # Feed surrogate covariance to Esprit algorithm
        doa_prediction, estimated_subspace = esprit(Rz, self.M, self.batch_size)
        eigen_values = torch.stack([pair[0] for pair in estimated_subspace])
        eigen_vectors = torch.stack([pair[1] for pair in estimated_subspace]) # shape: [N, D]
        cov_doa = doa_covariance_from_eig(eigen_values, eigen_vectors, doa_prediction, self.T)
        return Rz, doa_prediction

    def sense_device_forward(self, x):
        self.batch_size = x.shape[0]
        x_e = self.encoder_signal(x)
        # Quantize
        x_normalized = x_e - x_e.mean()
        # quantize if needed
        if self.quantize_source:
            x_normalized_real = torch.view_as_real(x_normalized)
            z_quantized, vq_loss = self.quantizer_signal(x_normalized_real)

            self.__unique_indices_set.update(torch.unique(z_quantized).tolist())
            self.codebook_utilization = len(self.__unique_indices_set) / self.codebook_size
            z_quantized = torch.view_as_complex(z_quantized)
        else:
            z_quantized, vq_loss = x_normalized, 0
        return vq_loss, z_quantized

    def calculate_cov_batch(self, x_batch):
        # Compute mean across sequence (dim=2)
        X_mean = x_batch.mean(dim=2, keepdim=True)

        # Center the data
        X_centered = x_batch - X_mean

        # Compute covariance in a batch-wise manner
        batch_cov_matrices = torch.matmul(X_centered, torch.conj(X_centered.transpose(1, 2))) / (
                    x_batch.shape[2] - 1)



        return batch_cov_matrices


    def init_online_history(self, horizon_constant):
        """
        This function introduce online infernece with init of time horizon and 
        Coveriance Matrix history holder.
        Args:
            horizon_constant: The time constant each samples quantized and then decoded

        Returns:
            Inatalization of the relvant field members
        """
        self.online_inference = True
        self.online_cov_history = []
        self.sub_horizon_time_constant = horizon_constant
        self.horizon_index = 0
        self.online_inference_factor = 0.5


    def calculate_progressive_coveriance(self, current_rx_matrix):
        """
        This function introduce online infernece with init of time horizon and calculate the progressive
        Coveriance matrix in each and every step.
        Args:
            current_rx_matrix:
                The coveriance matrix calculated for observed time horizon T'
        Returns:
            If this is the first step we return current rx matrix otherwise we return
            the combination of previous aggregated matrix and currently calculated matrix
        """
        if self.online_inference:
            if self.horizon_index == 0:
                self.online_cov_history.insert(self.horizon_index, current_rx_matrix)
                self.horizon_index += 1
            else:
                temp_k = self.horizon_index
                calculated_cov = (temp_k / (temp_k + 1)) * self.online_cov_history[self.horizon_index - 1] + (1 / (temp_k + 1)) * current_rx_matrix
                self.online_cov_history.insert(self.horizon_index, calculated_cov)
                self.horizon_index += 1
                return calculated_cov

        return current_rx_matrix


    def reset_online_history(self):
        """
        Reset the aggreative converiance history to deault values
        Returns:

        """
        self.online_cov_history = []
        self.horizon_index = 0


class TaskIgnorantSubspaceNet(SubspaceNetEsprit):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.
        SubspaceNetEsprit is based on the ability to perform back-propagation using ESPRIT algorithm,
        instead of RootMUSIC.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.

    """

    def __init__(self, N: int, T: int, tau: int, M: int, quantize_source=False, codebook_size=256):
        super().__init__(8, M)
        self.quantize_source = quantize_source

        self.N = N
        self.T = T

        in_channels = 8
        hidden_channels = 32
        out_channels = 8
        latent_dim=16

        # TODO: check sizes of Conv


        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.complex_rectifier = ComplexReLU(self.anti_rectifier)

        self.anti_rectifier_layer = AntiRectifierLayer(self.complex_rectifier)

        self.conv1 = ComplexConv1d(in_channels, 16, kernel_size=2)
        self.conv2 = ComplexConv1d(32, 32, kernel_size=2)
        self.conv3 = ComplexConv1d(64, 64, kernel_size=2)

        self.deconv2 = ComplexConvTranspose1d(128, 32, kernel_size=2)
        self.deconv3 = ComplexConvTranspose1d(64, 16, kernel_size=2)
        self.deconv4 = ComplexConvTranspose1d(32, out_channels, kernel_size=2)

        self.encoder_signal = nn.Sequential(self.conv1,
                                            # self.batchnorm1,
                                            self.anti_rectifier_layer,
                                            self.conv2,
                                            # self.batchnorm2,
                                            self.anti_rectifier_layer,
                                            self.conv3)


        self.DropOut = ComplexDropout(0.2)
        self.ReLU = nn.ReLU()

        # Define The decoder of the AE architecture
        self.decoder_signal = nn.Sequential(self.anti_rectifier_layer,
                                            self.deconv2,
                                            self.anti_rectifier_layer,
                                            self.deconv3,
                                            self.anti_rectifier_layer,
                                            self.DropOut,
                                            self.deconv4)

        num_embeddings = 4
        self.codebook_size = codebook_size
        lambda_c = 0.1
        lambda_p = 0.33
        self.quantizer_signal = FixedVectorQuantizer(num_embeddings, self.codebook_size, lambda_c, lambda_p)

        self.__unique_indices_set = set()

    def set_quantize(self, quantize: bool):
        self.quantize_source = quantize

    def forward(self, x: torch.Tensor):
        self.batch_size = x.shape[0]

        x_e = self.encoder_signal(x)

        # Quantize
        x_normalized = x_e - x_e.mean()

        # quantize if needed
        if self.quantize_source:
            x_normalized_real = torch.view_as_real(x_normalized)
            z_quantized, vq_loss = self.quantizer_signal(x_normalized_real)

            self.__unique_indices_set.update(torch.unique(z_quantized).tolist())
            self.codebook_utilization = len(self.__unique_indices_set) / self.codebook_size
            z_quantized = torch.view_as_complex(z_quantized)
        else:
            z_quantized, vq_loss = x_normalized, 0

        x_hat = self.decoder_signal(z_quantized)

        # calculate loss over restoration and vq_loss
        reconstruction_loss = torch.nn.functional.mse_loss(torch.view_as_real(x_hat), torch.view_as_real(x))
        total_loss = reconstruction_loss + vq_loss

        # Create the auto-correlation matrix out of after the quantization
        Rx_matrix = self.calculate_cov_batch(x_hat)

        # Apply Gram operation diagonal loading
        Rz = add_epsilon_batch(Kx=Rx_matrix, eps=1, batch_size=self.batch_size)

        # Feed surrogate covariance to the differentiable subspace algorithm
        """method_output = self.diff_method(Rz, self.M, self.batch_size)
        if isinstance(method_output, tuple):
            # Root MUSIC output
            doa_prediction, doa_all_predictions, roots = method_output
        else:
            # Esprit output
            doa_prediction = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz, total_loss
        """
        doa_prediction, estimated_subspace = esprit(Rz, self.M, self.batch_size)
        return doa_prediction, Rz, total_loss


    def calculate_cov_batch(self, x_batch):
        # Compute mean across sequence (dim=2)
        X_mean = x_batch.mean(dim=2, keepdim=True)

        # Center the data
        X_centered = x_batch - X_mean

        # Compute covariance in a batch-wise manner
        batch_cov_matrices = torch.matmul(X_centered, torch.conj(X_centered.transpose(1, 2))) / (
                x_batch.shape[2] - 1)

        return batch_cov_matrices



class DeepAugmentedMUSIC(nn.Module):
    """DeepAugmentedMUSIC is a model-based deep learning model for Direction of Arrival (DOA) estimation.

    Attributes:
        N (int): Number of sensors.
        T (int): Number of observations.
        M (int): Number of sources.
        angels (torch.Tensor): Tensor containing angles from -pi/2 to pi/2.
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden layer.
        rnn (nn.GRU): Recurrent neural network module.
        fc (nn.Linear): Fully connected layer.
        fc1 (nn.Linear): Fully connected layer.
        fc2 (nn.Linear): Fully connected layer.
        fc3 (nn.Linear): Fully connected layer.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.
        DropOut (nn.Dropout): Dropout layer.
        BatchNorm (nn.BatchNorm1d): Batch normalization layer.
        sv (torch.Tensor): Steering vector.

    Methods:
        steering_vec(): Computes the steering vector based on the specified parameters.
        spectrum_calculation(Un: torch.Tensor): Calculates the MUSIC spectrum.
        pre_MUSIC(Rz: torch.Tensor): Applies the MUSIC operation for generating the spectrum.
        forward(X: torch.Tensor): Performs the forward pass of the DeepAugmentedMUSIC model.
    """

    def __init__(self, N: int, T: int, M: int):
        """Initializes the DeepAugmentedMUSIC model.

        Args:
        -----
            N (int): Number of sensors.
            M (int): Number of sources.
            T (int): Number of observations.
        """
        super(DeepAugmentedMUSIC, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * np.pi / 2, np.pi / 2, 361)
        self.input_size = 2 * self.N
        self.hidden_size = 2 * self.N
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size * self.N)
        self.fc1 = nn.Linear(self.angels.shape[0], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.M)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(0.25)
        self.BatchNorm = nn.BatchNorm1d(self.T)
        self.sv = self.steering_vec()
        # Weights initialization
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)

    def steering_vec(self):
        """Computes the ideal steering vector based on the specified parameters.
            equivalent to src.system_model.steering_vec method, but support pyTorch.

        Returns:
        --------
            tensor.Torch: the steering vector
        """
        sv = []
        for angle in self.angels:
            a = torch.exp(
                -1 * 1j * np.pi * torch.linspace(0, self.N - 1, self.N) * np.sin(angle)
            )
            sv.append(a)
        return torch.stack(sv, dim=0)

    def spectrum_calculation(self, Un: torch.Tensor):
        spectrum_equation = []
        for i in range(self.angels.shape[0]):
            spectrum_equation.append(
                torch.real(
                    torch.conj(self.sv[i]).T @ Un @ torch.conj(Un).T @ self.sv[i]
                )
            )
        spectrum_equation = torch.stack(spectrum_equation, dim=0)
        spectrum = 1 / spectrum_equation

        return spectrum, spectrum_equation

    def pre_MUSIC(self, Rz: torch.Tensor):
        """Applies the MUSIC operration for generating spectrum

        Args:
            Rz (torch.Tensor): Generated covariance matrix

        Returns:
            torch.Tensor: The generated MUSIC spectrum
        """
        spectrum = []
        bs_Rz = Rz
        for iter in range(self.batch_size):
            R = bs_Rz[iter]
            # Extract eigenvalues and eigenvectors using EVD
            _, eigenvectors = torch.linalg.eig(R)
            # Noise subspace as the eigenvectors which associated with the M first eigenvalues
            Un = eigenvectors[:, self.M :]
            # Calculate MUSIC spectrum
            spectrum.append(self.spectrum_calculation(Un)[0])
        return torch.stack(spectrum, dim=0)

    def forward(self, X: torch.Tensor):
        """
        Performs the forward pass of the DeepAugmentedMUSIC model.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: The estimated DOA.
        """
        # X shape == [Batch size, N, T]
        self.BATCH_SIZE = X.shape[0]
        ## Architecture flow ##
        # decompose X and concatenate real and imaginary part
        X = torch.cat(
            (torch.real(X), torch.imag(X)), 1
        )  # Shape ==  [Batch size, 2N, T]
        # Reshape Output shape: [Batch size, T, 2N]
        X = X.view(X.size(0), X.size(2), X.size(1))
        # Apply batch normalization
        X = self.BatchNorm(X)
        # GRU Clock
        gru_out, hn = self.rnn(X)
        Rx = gru_out[:, -1]
        # Reshape Output shape: [Batch size, 1, 2N]
        Rx = Rx.view(Rx.size(0), 1, Rx.size(1))
        # FC Block
        Rx = self.fc(Rx)  # Shape: [Batch size, 1, 2N^2])
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_view = Rx.view(self.BATCH_SIZE, 2 * self.N, self.N)
        Rx_real = Rx_view[:, : self.N, :]  # Shape [Batch size, N, N])
        Rx_imag = Rx_view[:, self.N :, :]  # Shape [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape [Batch size, N, N])
        # Build MUSIC spectrum
        spectrum = self.pre_MUSIC(Kx_tag)  # Shape [Batch size, 361(grid_size)])
        # Apply peak detection using FC block #2
        y = self.ReLU(self.fc1(spectrum))  # Shape [Batch size, 361(grid_size)])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N)
        # Find doa
        DOA = self.fc3(y)  # Shape [Batch size, M)
        return DOA


class DeepCNN(nn.Module):
    """DeepCNN is a convolutional neural network model for DoA  estimation.

    Args:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.

    Attributes:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        fc1 (nn.Linear): Fully connected layer 1.
        BatchNorm (nn.BatchNorm2d): Batch normalization layer.
        fc2 (nn.Linear): Fully connected layer 2.
        fc3 (nn.Linear): Fully connected layer 3.
        fc4 (nn.Linear): Fully connected layer 4.
        DropOut (nn.Dropout): Dropout layer.
        Sigmoid (nn.Sigmoid): Sigmoid activation function.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.

    Methods:
        forward(X: torch.Tensor): Performs the forward pass of the DeepCNN model.
    """

    def __init__(self, N, grid_size):
        ## input dim (N, T)
        super(DeepCNN, self).__init__()
        self.N = N
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.fc1 = nn.Linear(256 * (self.N - 5) * (self.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)
        self.DropOut = nn.Dropout(0.3)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, X):
        # X shape == [Batch size, N, N, 3]
        X = X.view(X.size(0), X.size(3), X.size(2), X.size(1))  # [Batch size, 3, N, N]
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.ReLU(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.ReLU(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))  # [Batch size, 4096]
        X = self.DropOut(self.ReLU(self.fc2(X)))  # [Batch size, 2048]
        X = self.DropOut(self.ReLU(self.fc3(X)))  # [Batch size, 1024]
        X = self.fc4(X)  # [Batch size, grid_size]
        X = self.Sigmoid(X)
        return X


def root_music(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent tosrc.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """

    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    subspace_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)

        # Add subspace information to later use in uncertanty
        subspace_batches.append((eigenvalues, eigenvectors))

        # Assign noise subspace as the eigenvectors associated with M greatest eigenvalues
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        # Generate hermitian noise subspace matrix
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        # Calculates the sum of F matrix diagonals
        diag_sum = sum_of_diags_torch(F)
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = find_roots_torch(diag_sum)
        # Calculate the phase component of the roots
        roots_angels_all = torch.angle(roots)
        # Calculate doa
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        # Take only roots which inside the unit circle
        roots = roots[
            sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))
        ]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        # Calculate the phase component of the roots
        roots_angels = torch.angle(roots)
        # Calculate doa
        doa_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)

    return (
        torch.stack(doa_batches, dim=0),
        torch.stack(doa_all_batches, dim=0),
        roots_to_return,
    ), subspace_batches


def esprit(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Esprit algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent to src.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
    """

    doa_batches = []

    subspace_batches = []

    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)

        subspace_batches.append((eigenvalues, eigenvectors))

        # Get signal subspace
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M]
        # Separate the signal subspace into 2 overlapping subspaces
        Us_upper, Us_lower = (
            Us[0 : R.shape[0] - 1],
            Us[1 : R.shape[0]],
        )
        # Generate Phi matrix
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        # Find eigenvalues and eigenvectors (EVD) of Phi
        phi_eigenvalues, _ = torch.linalg.eig(phi)
        # Calculate the phase component of the roots
        eigenvalues_angels = torch.angle(phi_eigenvalues)
        # Calculate the DoA out of the phase component
        doa_predictions = -1 * torch.arcsin((1 / np.pi) * eigenvalues_angels)

        #TODO: Check for convension
        #phase_shifts = torch.angle(phi_eigenvalues)
        #sin_theta = torch.clamp(phase_shifts / np.pi, -1.0, 1.0)
        #doa_predictions = torch.arcsin(sin_theta)

        doa_batches.append(doa_predictions)

    return torch.stack(doa_batches, dim=0), subspace_batches

## We introduce measure of uncertanty from
## Asymptotic Performance Analysis of ESPRIT, Higher Order ESPRIT, and Virtual ESPRIT Algorithms
## By Norman Yuen and Benjamin Friedlander,

def steering_vec(N, theta, d=0.5, device=None):
    """ULA steering vector  a(θ)  with spacing d·λ."""
    n = torch.arange(N, device=device)
    return torch.exp(1j * math.pi * d * n * torch.sin(theta))

def steering_deriv(N, theta, d=0.5, device=None):
    """∂a/∂θ for a ULA."""
    n = torch.arange(N, device=device)
    # Maybe later to use a precomputational steering vector
    return 1j * math.pi * d * n * torch.cos(theta) * steering_vec(N, theta, d, device)

def doa_covariance_from_eig(evals, evecs, thetas, N, d=0.5):
    """
    evals  : (B, M)              eigenvalues  (complex)
    evecs  : (B, M, M)           eigenvectors (columns)
    thetas : (B, r)  or  (r,)    DoA estimates  (rad)
    N      : scalar or (B,)      snapshots used for R̂
    d      : element spacing in λ units (default 0.5)
    returns: (B, r, r)           plug-in CRB  Σ̂_θ
    """
    B, M = evals.shape
    device = evals.device
    r = thetas.shape[-1]              #  ------------  A

    # 1. sort eigen-pairs
    idx = torch.argsort(evals.real, dim=-1, descending=True)
    evecs_sorted = torch.gather(evecs, 2, idx.unsqueeze(1).expand(-1, M, -1))
    evals_sorted = torch.gather(evals, 1, idx)

    # 2. noise sub-space  U_n   and projector  P_perp
    Un  = evecs_sorted[:, :, r:]           # (B, M, M−r)
    Un  = torch.linalg.qr(Un).Q           #  ------------  C  (B, M, M−r)
    I   = torch.eye(M, device=device).expand(B, -1, -1)
    Pperp = I - Un @ Un.conj().transpose(-1, -2)   #  ----  B (B,M,M)


    Pperp = 0.5 * (Pperp + Pperp.conj().transpose(-1, -2))  # enforce Hermitian

    # 3. noise power σ̂²   (mean of noise eigenvalues)
    sigma2 = evals_sorted[:, r:].real.mean(dim=-1)      # (B,)

    # 4. steering matrix  A(θ̂)  and derivative  D(θ̂)
    if thetas.ndim == 1:
        thetas = thetas.unsqueeze(0).repeat(B, 1)       # (B, r)
    m = torch.arange(M, device=device).view(1, M, 1)    # (1,M,1)
    sin_t = torch.sin(thetas).view(B, 1, r)
    cos_t = torch.cos(thetas).view(B, 1, r)
    A = torch.exp(1j * math.pi * d * m * sin_t)         # (B,M,r)
    D = 1j * math.pi * d * m * cos_t * A                # (B,M,r)

    # 5. Fisher information matrix  G = Dᴴ P⊥ D
    G = D.conj().transpose(-1, -2) @ Pperp @ D          # (B,r,r)

    # -----------------------------------------------
    #
    G = 0.5 * (G + G.conj().transpose(-1, -2))
    eps = torch.linalg.eigvalsh(G).real.amax(dim=-1, keepdim=True) * 1e-6
    Ginv = torch.linalg.inv(G + eps.unsqueeze(-1) * torch.eye(r, device=G.device))
    # -----------------------------------------------
    # Yuen & Friedlander (eq. ★)
    if torch.is_tensor(N):
        N = N.to(device).view(B, 1, 1)

    Sigma = (sigma2 / (2 * N)).view(B, 1, 1) * torch.real(Ginv)
    return Sigma



def empirical_error_cov(theta_hat: torch.Tensor,
                        theta_true: torch.Tensor) -> torch.Tensor:
    """
    Empirical covariance of the estimation error e = θ̂ − θ.

    Parameters
    ----------
    theta_hat : (..., K) tensor       -- estimated DoA(s) - deg
    theta_true: (..., K) tensor       -- ground-truth DoA(s) - deg
        ▸ The leading dimension(s) ‘...’ index independent experiments (N).
        ▸ K is the number of parameters per experiment
          (K = 1  → single source;  K > 1 → multi-source or multi-dim).

    Returns
    -------
    Σ̂ : (K, K) covariance matrix  (deg² or rad², same units as inputs)
          For K = 1 this collapses to a scalar variance.

    Notes
    -----
    • Errors wrap around at ±180° for angles in degrees, so :

        e = ( (theta_hat - theta_true + 180) % 360 ) - 180   # wrap to (-180,180]

    • Unbiased divisor (N-1) is used.
    """
    # 1. compute error
    #e = ((theta_hat - theta_true + 180) % 360) - 180        # shape (..., K)
    e = theta_hat - theta_true

    # 2. flatten leading dims → (N, K)
    if e.ndim == 1:                   # fast path: scalar DoA
        return torch.var(e, unbiased=True)

    #N = e.shape[0] = e.view(-1, e.shape[-1])  # (N, K)

    # 3. centre & accumulate cross-products
    e_centered = e - e.mean(dim=0, keepdim=True)
    Σ_hat = e_centered.t().conj() @ e_centered / (e.shape[0] - 1)
    return Σ_hat
