"""Subspace-Net 
Details
----------
Name: criterions.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 03/06/23

Purpose:
--------
The purpose of this script is to define and document several loss functions (RMSPELoss and MSPELoss)
and a helper function (permute_prediction) for calculating the Root Mean Square Periodic Error (RMSPE)
and Mean Square Periodic Error (MSPE) between predicted values and target values.
The script also includes a utility function RMSPE and MSPE that calculates the RMSPE and MSPE values
for numpy arrays.

This script includes the following Classes anf functions:

* permute_prediction: A function that generates all possible permutations of a given prediction tensor.
* RMSPELoss (class): A custom PyTorch loss function that calculates the RMSPE loss between predicted values
    and target values. It inherits from the nn.Module class and overrides the forward method to perform
    the loss computation.
* MSPELoss (class): A custom PyTorch loss function that calculates the MSPE loss between predicted values
  and target values. It inherits from the nn.Module class and overrides the forward method to perform the loss computation.
* RMSPE (function): A function that calculates the RMSPE value between the DOA predictions and target DOA values for numpy arrays.
* MSPE (function): A function that calculates the MSPE value between the DOA predictions and target DOA values for numpy arrays.
* set_criterions(function): Set the loss criteria based on the criterion name.

"""

import numpy as np
import torch.nn as nn
import torch
from itertools import permutations

import torch.nn.functional as F
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])
        
    """
    torch_perm_list = []
    for p in list(permutations(range(prediction.shape[0]),prediction.shape[0])):
        torch_perm_list.append(prediction.index_select( 0, torch.tensor(list(p), dtype = torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim = 0)
    return predictions

class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self):
        super(RMSPELoss, self).__init__()
    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            prediction_perm = permute_prediction(batch_predictions).to(device)
            for prediction in prediction_perm:
                # Calculate error with modulo pi
                error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
                # Calculate RMSE over all permutations
                rmspe_val = (1 / np.sqrt(len(targets))) * torch.linalg.norm(error)
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim = 0)
            # Choose minimal error from all permutations
            rmspe_min = torch.min(rmspe_tensor)
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim = 0))
        return result


class MultiRMSPELoss(nn.Module):
    """ Multi-Subarray Root Mean Square Periodic Error (RMSPE) loss function.
    Calculates the RMSPE for each subarray completely independently,
    matching subarray predictions strictly to that specific subarray's targets.

    Args:
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, reduction='mean'):
        super(MultiRMSPELoss, self).__init__()
        self.reduction = reduction

    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor):
        """
        Args:
            doa_predictions (torch.Tensor): Shape (batch_size, num_subarrays, num_predictions)
            doa (torch.Tensor): Shape (batch_size, num_subarrays, num_targets)

        Returns:
            torch.Tensor: The computed RMSPE loss.
        """
        device = doa_predictions.device
        B, S, P = doa_predictions.shape  # Batch, Subarrays, Predictions

        # 1. Generate permutation indices for the predictions
        # Shape: (num_permutations, P)
        perm_indices = torch.tensor(list(itertools.permutations(range(P))), device=device)
        num_perms = perm_indices.shape[0]

        # 2. To strictly isolate subarrays, we expand our tensors to 4 Dimensions:
        # [Batch, Subarray, Permutations, Predictions]

        # Expand predictions so we have a copy for every permutation
        preds_exp = doa_predictions.unsqueeze(2).expand(B, S, num_perms, P)

        # Expand permutation indices to match the 4D shape
        perms_exp = perm_indices.view(1, 1, num_perms, P).expand(B, S, num_perms, P)

        # 3. Apply the permutations INDEPENDENTLY to every subarray using torch.gather
        # Shape remains: [B, S, num_perms, P]
        preds_perm = torch.gather(preds_exp, dim=3, index=perms_exp)

        # 4. Expand the ground truth targets to match the 4D shape
        # Shape: [B, S, 1, P] -> Broadcasting will compare targets to all permutations
        targets_exp = doa.unsqueeze(2)

        # 5. Calculate wrapped angular error modulo pi
        error = (((preds_perm - targets_exp) + (np.pi / 2)) % np.pi) - np.pi / 2

        # 6. Calculate RMSPE over the Prediction dimension (dim=3)
        # Shape becomes: [B, S, num_perms]
        rmspe_vals = (1 / np.sqrt(P)) * torch.linalg.norm(error, dim=3)

        # 7. Find the minimal error permutation for EACH Subarray (dim=2)
        # Shape becomes: [B, S]
        rmspe_min = torch.min(rmspe_vals, dim=2)[0]

        # 8. Apply the final reduction over the batch and subarrays
        if self.reduction == 'mean':
            return torch.mean(rmspe_min)
        elif self.reduction == 'sum':
            return torch.sum(rmspe_min)
        else:
            return rmspe_min  # Returns exact loss per subarray [B, S]



class UEELoss(nn.Module):
    """ Uncertainty Empirical Error loss function for Per-Source Uncertainty.
    Calculates the L2 loss between predicted per-source variance and empirical per-source squared error.

    Args:
        reduction (str): Specifies the reduction to apply to the output: 'mean' or 'sum'. Default: 'sum'.
    """

    def __init__(self, reduction='mean'):
        super(UEELoss, self).__init__()
        self.reduction = reduction

    def forward(self, uncertainty_prediction: torch.Tensor, doa_predictions: torch.Tensor,
                doa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            uncertainty_prediction (torch.Tensor): Predicted variance tensor [batch_size, num_predictions].
            doa_predictions (torch.Tensor): Predicted DOA values tensor [batch_size, num_predictions].
            doa (torch.Tensor): Target values tensor [batch_size, num_targets].

        Returns:
            torch.Tensor: The computed Uncertainty Empirical Error loss.
        """
        device = doa_predictions.device
        batch_size, num_preds = doa_predictions.shape

        # Generate all permutation indices once for the predictions
        # Shape: [num_permutations, num_preds]
        # This replaces the custom `permute_prediction` function to ensure safe alignment
        perm_indices = torch.tensor(list(itertools.permutations(range(num_preds))), device=device)

        uee_loss_list = []

        for iter in range(batch_size):
            batch_preds = doa_predictions[iter]  # Shape: [num_preds]
            batch_uncert = uncertainty_prediction[iter]  # Shape: [num_preds]
            targets = doa[iter]  # Shape: [num_preds]

            # Apply the permutation indices to BOTH DOAs and Uncertainties
            # Shapes become: [num_permutations, num_preds]
            prediction_perms = batch_preds[perm_indices]
            uncertainty_perms = batch_uncert[perm_indices]

            # 1. Calculate wrapped angular error for all permutations simultaneously
            error = (((prediction_perms - targets) + (np.pi / 2)) % np.pi) - np.pi / 2

            # 2. Find the best assignment (permutation) based on DOA error
            # We calculate the Mean Squared Error of the DOAs for each permutation
            doa_mse_per_perm = torch.mean(error ** 2, dim=1)

            # Get the index of the permutation that best aligns predictions with targets
            best_perm_idx = torch.argmin(doa_mse_per_perm)

            # 3. Extract the optimal alignment for error and uncertainty
            best_error = error[best_perm_idx]  # Shape: [num_preds]
            best_uncertainty = uncertainty_perms[best_perm_idx]  # Shape: [num_preds]

            # 4. Calculate empirical variance per source (error squared)
            empirical_variance = best_error ** 2

            # 5. Calculate L2 loss between predicted per-source uncertainty and empirical variance
            uee = F.mse_loss(best_uncertainty, empirical_variance, reduction='mean')

            uee_loss_list.append(uee)

        uee_tensor = torch.stack(uee_loss_list, dim=0)

        # Apply the chosen batch reduction method
        if self.reduction == 'mean':
            return torch.mean(uee_tensor)
        else:
            return torch.sum(uee_tensor)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools


class CombinedUncertaintyLoss(nn.Module):
    """
    Combined Accuracy and Reliability Loss for Multiple Subarrays.

    Calculates the loss by ACCUMULATING over the sources, and then averaging over the batch:
    UE = mean_dataset[ sum_sources( (predicted_variance - empirical_error^2)^2 ) ]

    Args:
        lambda_val (float): Weighting factor between [0, 1].
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, lambda_val=0.7, reduction='mean'):
        super(CombinedUncertaintyLoss, self).__init__()
        self.lambda_val = lambda_val
        self.reduction = reduction

    def forward(self, uncertainty_predictions: torch.Tensor, doa_predictions: torch.Tensor, doa: torch.Tensor):
        device = doa_predictions.device
        B, S, P = doa_predictions.shape  # Batch, Subarrays, Predictions (Sources)

        # 1. Generate permutation indices for the predictions
        perm_indices = torch.tensor(list(itertools.permutations(range(P))), device=device)
        num_perms = perm_indices.shape[0]

        # 2. Expand tensors to 4D to isolate subarrays and permutations: [B, S, num_perms, P]
        preds_exp = doa_predictions.unsqueeze(2).expand(B, S, num_perms, P)
        uncert_exp = uncertainty_predictions.unsqueeze(2).expand(B, S, num_perms, P)
        perms_exp = perm_indices.view(1, 1, num_perms, P).expand(B, S, num_perms, P)

        # 3. Apply permutations strictly to predictions and uncertainties using gather
        preds_perm = torch.gather(preds_exp, dim=3, index=perms_exp)
        uncert_perm = torch.gather(uncert_exp, dim=3, index=perms_exp)

        # 4. Expand ground truth targets for broadcasting: [B, S, 1, P]
        targets_exp = doa.unsqueeze(2)

        # 5. Calculate wrapped angular error modulo pi for all permutations
        error = (((preds_perm - targets_exp) + (np.pi / 2)) % np.pi) - np.pi / 2

        # 6. Find the optimal alignment (permutation) for the subarray (Global 1-to-1 matching)
        rmspe_squared_all = torch.mean(error ** 2, dim=3)
        best_perm_idx = torch.argmin(rmspe_squared_all, dim=2, keepdim=True)

        # Expand index to gather the matching errors and uncertainties: [B, S, 1, P]
        best_perm_idx_exp = best_perm_idx.unsqueeze(3).expand(B, S, 1, P)

        # ---------------------------------------------------------
        # PER-SOURCE LOSS CALCULATION
        # ---------------------------------------------------------

        # Extract optimal error and uncertainty: Shapes [B, S, P]
        best_error = torch.gather(error, dim=2, index=best_perm_idx_exp).squeeze(2)
        best_uncert = torch.gather(uncert_perm, dim=2, index=best_perm_idx_exp).squeeze(2)

        # A. Accuracy Loss Per Source (Squared Error)
        acc_loss_per_source = best_error ** 2

        # B. Uncertainty Loss Per Source (Variance vs Empirical Squared Error)
        predicted_variance = best_uncert ** 2

        # Explicitly written exactly as your formula: (pred_var - error^2)^2
        ue_loss_per_source = (predicted_variance - acc_loss_per_source) ** 2

        # C. Combine Loss Per Source
        combined_loss_per_source = (self.lambda_val * acc_loss_per_source) + (
                    (1.0 - self.lambda_val) * ue_loss_per_source)

        # ---------------------------------------------------------
        # THE FIX: ACCUMULATE OVER SOURCES (Sum across dim=2)
        # ---------------------------------------------------------
        # Shapes become: [Batch, Subarrays]
        combined_loss_accumulated = torch.sum(combined_loss_per_source, dim=2)
        acc_loss_accumulated = torch.sum(acc_loss_per_source, dim=2)
        ue_loss_accumulated = torch.sum(ue_loss_per_source, dim=2)

        # ---------------------------------------------------------
        # REDUCTION OVER THE DATASET / BATCH
        # ---------------------------------------------------------
        if self.reduction == 'mean':
            return (
                torch.mean(combined_loss_accumulated),
                torch.mean(acc_loss_accumulated),
                torch.mean(ue_loss_accumulated)
            )
        elif self.reduction == 'sum':
            return (
                torch.sum(combined_loss_accumulated),
                torch.sum(acc_loss_accumulated),
                torch.sum(ue_loss_accumulated)
            )
        else:
            return combined_loss_accumulated, acc_loss_accumulated, ue_loss_accumulated

class MSPELoss(nn.Module):
    """Mean Square Periodic Error (MSPE) loss function.
    This loss function calculates the MSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the MSPE loss between the predictions and target values.

    Example:
        criterion = MSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """
    def __init__(self):
        super(MSPELoss, self).__init__()
    def forward(self, doa_predictions: torch.Tensor, doa):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).

        Returns:
            torch.Tensor: The computed MSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            prediction_perm = permute_prediction(batch_predictions).to(device)
            for prediction in prediction_perm:
                # Calculate error with modulo pi
                error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
                # Calculate MSE over all permutations
                rmspe_val = (1 / len(targets)) * (torch.linalg.norm(error) ** 2)
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim = 0)
            rmspe_min = torch.min(rmspe_tensor)
            # Choose minimal error from all permutations
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim = 0))
        return result

def RMSPE(doa_predictions: np.ndarray, doa: np.ndarray):
    """
    Calculate the Root Mean Square Periodic Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    rmspe_list = []
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = np.array(p)
        doa = np.array(doa)
        # Calculate error with modulo pi
        error = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        # Calculate RMSE over all permutations
        rmspe_val = (1 / np.sqrt(len(p))) * np.linalg.norm(error)
        rmspe_list.append(rmspe_val)
    # Choose minimal error from all permutations
    return np.min(rmspe_list)

def MSPE(doa_predictions: np.ndarray, doa: np.ndarray):
    """Calculate the Mean Square Percentage Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    rmspe_list = []
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = np.array(p)
        doa = np.array(doa)
        # Calculate error with modulo pi
        error = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        # Calculate MSE over all permutations
        rmspe_val = (1 / len(p)) * (np.linalg.norm(error) ** 2)
        rmspe_list.append(rmspe_val)
    # Choose minimal error from all permutations
    return np.min(rmspe_list)


def MSPE_Empricial(doa_predictions: np.ndarray, doa: np.ndarray):
    """
    This function calculate the Empricial MSPE between the DOA predictions and target DOA values.
    :param doa_predictions:
    :param doa:
    :return: Empricial MSPE value in degrees.
    """
    rmspe_list = []
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = np.array(p)
        doa = np.array(doa)
        # Calculate error with modulo pi
        error = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        # Calculate MSE over all permutations
        rmspe_val = (np.linalg.norm(np.rad2deg(error)) ** 2)
        rmspe_list.append(rmspe_val)
    # Choose minimal error from all permutations
    return np.min(rmspe_list)

def BPE(doa_predictions: np.ndarray, doa: np.ndarray):
    """Calculate the Mean Square Percentage Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    bias_list = []
    for p in list(permutations(doa_predictions, len(doa_predictions))):
        p = np.array(p)
        doa = np.array(doa)
        # Calculate error with modulo pi
        bias = (((p - doa) * np.pi / 180) + np.pi / 2) % np.pi - np.pi / 2
        # Calculate MSE over all permutations
        bias_p =  bias

        bias_list.append(np.abs(np.rad2deg(bias_p)))
    # Choose minimal error from all permutations
    return np.min(bias_list)



def set_criterions(criterion_name:str):
    """
    Set the loss criteria based on the criterion name.

    Parameters:
        criterion_name (str): Name of the criterion.

    Returns:
        criterion (nn.Module): Loss criterion for model evaluation.
        subspace_criterion (Callable): Loss criterion for subspace method evaluation.

    Raises:
        Exception: If the criterion name is not defined.
    """
    if criterion_name.startswith("rmse"):
        criterion = RMSPELoss()
        subspace_criterion = RMSPE
    elif criterion_name.startswith("mse"):
        criterion = MSPELoss()
        subspace_criterion = MSPE
    else:
        raise Exception(f"criterions.set_criterions: Criterion {criterion_name} is not defined")
    print(f"Loss measure = {criterion_name}")
    return criterion, subspace_criterion

if __name__ == "__main__":
    prediction = torch.tensor([1, 2, 3])
    print(permute_prediction(prediction))