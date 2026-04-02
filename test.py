import sys
import os
import math
from signal import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from trainer import graph_scene_collate
from src.multi_subarrays_model import MultiSubarraysModel
from src.multi_model_dataset import SensorSourceGraphDataset

from src.criterions import RMSPELoss, CombinedUncertaintyLoss

from src.learned_agg_layer import match_learned_attn_shapes
from src.localization_block import position_errors

from src.visualization import visualize_ray_frame
from src.uncertainty_block import plot_sigma_vs_doa

from torch.profiler import (
    profile,
    schedule,
    tensorboard_trace_handler,
    ProfilerActivity,
    record_function,
)

from src.uncertainty_block import UncertaintyEstimation


def calculate_true_uncertainty(doa_true_rad, samples, plot=False):
    """
    Calculates the 'true' analytical uncertainty using the covariance matrix Rx.
    Returns the tensor so it can be compared against the network's predictions.
    """
    device = doa_true_rad.device
    iq_samples = samples.squeeze(dim=2)
    signal_shape_uncertainty = samples.shape[-1]

    uncertainty_block = UncertaintyEstimation(signal_shape=signal_shape_uncertainty).to(device)

    # Dynamically grab the number of sources (removes hardcoded M=2)
    num_sources = doa_true_rad.shape[-1]

    Rx_batch = []
    for batch_index in range(iq_samples.shape[0]):
        Rx = []
        for subarray_index in range(iq_samples.shape[1]):
            Rx.append(torch.cov(iq_samples[batch_index, subarray_index, :, :]))
        Rx_batch.append(torch.stack(Rx))

    RX_batch = torch.stack(Rx_batch).to(device)
    sigma_true = torch.zeros(iq_samples.shape[0], iq_samples.shape[1], num_sources, device=device)

    for subarray_index in range(iq_samples.shape[1]):
        sigma_true[:, subarray_index, :] = uncertainty_block.forward(
            torch.rad2deg(doa_true_rad[:, subarray_index, :]),
            Rx=RX_batch[:, subarray_index, :, :]
        )

    if plot:
        for index in range(doa_true_rad.shape[1]):
            plot_sigma_vs_doa(
                torch.rad2deg(doa_true_rad[:, index, :]).cpu(),
                sigma_true[:, index, :].cpu(),
                title="Sigma (with True Rx) vs True DoA"
            )

    return sigma_true


@torch.no_grad()
def evaluate(model, loader, criterion, device, batch_size, doa_only, profiler=None):
    model.eval()

    total_total_loss = 0.0
    total_rmspe_sq = 0.0
    total_ue_loss = 0.0
    total_sigma_mae = 0.0  # Track the error between pred and true sigma
    num_sources = 1

    for step, (sensor_positions, source_positions, samples, doa_gt) in enumerate(loader):
        with record_function("eval_step"):
            sensor_positions = sensor_positions.to(device)
            source_positions = source_positions.to(device)
            samples = samples.to(device)

            doa_gt = torch.deg2rad(doa_gt).to(device)
            num_sources = doa_gt.shape[-1]

            if doa_only:
                with record_function("model_forward_doa"):
                    model_result = model(sensor_positions, samples, doa_gt)

                    doa_pred = model_result["bearings"]
                    sigma_pred_deg = model_result["sigma_i"]

                    # ---------------------------------------------------------
                    # COMPARE PREDICTED SIGMA vs TRUE SIGMA
                    # ---------------------------------------------------------
                    # Calculate true sigma from Rx (Disable plotting for speed, or set to True)
                    sigma_true_deg = calculate_true_uncertainty(doa_gt, samples, plot=False)

                    # Calculate Mean Absolute Error between network prediction and true calculation
                    # Both are in degrees at this point
                    sigma_mae = F.l1_loss(sigma_pred_deg, sigma_true_deg)
                    total_sigma_mae += sigma_mae.item()

                    if model.estimate_position is True:
                        pos_pred = model_result["source_estimated_position"]
                        if model.estimate_uncertainty is True:
                            position_metrics = position_errors(
                                model_result["source_estimated_position"],
                                model_result["source_estimated_position_wls"],
                                source_positions
                            )
                            model_result["position_metrics"] = position_metrics
                    else:
                        pos_pred = torch.zeros_like(source_positions)

                    # Vectorized Loss Calculation
                    sigma_pred_rad = torch.deg2rad(sigma_pred_deg)
                    loss, rmspe_sq, ue_loss = criterion(sigma_pred_rad, doa_pred, doa_gt)

                    total_total_loss += loss.item()
                    total_rmspe_sq += rmspe_sq.item()
                    total_ue_loss += ue_loss.item()

                    # Visualization
                    for sample_index in range(min(10, sensor_positions.shape[0])):
                        visualize_ray_frame(
                            positions=sensor_positions[sample_index],
                            bearings=doa_pred[sample_index],
                            x_hat=pos_pred[sample_index],
                            x_true=source_positions[sample_index],
                            sigmas=torch.deg2rad(sigma_pred_deg[sample_index]),
                            step=step,
                            save_path=f"visualizations/sample_{sample_index:03d}_test_{step:03d}.png"
                        )

            else:
                with record_function("model_forward_pos"):
                    pred = model(sensor_positions, samples, None)
                loss = criterion(pred, source_positions)
                total_total_loss += loss.item()

        if profiler is not None:
            profiler.step()

    divisor = len(loader)
    return {
        "total_loss": total_total_loss / divisor,
        "rmspe_sq": total_rmspe_sq / divisor,
        "ue_loss": total_ue_loss / divisor,
        "sigma_mae": total_sigma_mae / divisor,  # Return the new metric
        "num_sources": num_sources
    }


def main(profiler=None):
    parser = argparse.ArgumentParser("Test Multi-Subarrays Model")
    parser.add_argument("--test_dataset_path", required=True, help="Path to test dataset .pt file")
    parser.add_argument("--config_path", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--train_doa_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_to_wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds: SensorSourceGraphDataset = torch.load(args.test_dataset_path, weights_only=False)
    test_ds.set_use_graph_features(use_features=True)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_scene_collate,
        drop_last=False,
    )

    sensor_positions = test_ds.get_sensor_potision()

    model = MultiSubarraysModel(
        sensors_positions=sensor_positions,
        multi_model_configuration=args.config_path,
        args=args
    ).to(device)

    model.estimate_uncertainty = True
    model.estimate_position = True

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    match_learned_attn_shapes(model, checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.train_doa_only:
        criterion = CombinedUncertaintyLoss(lambda_val=0.5, reduction='mean')
    else:
        criterion = torch.nn.MSELoss()

    if args.log_to_wandb:
        wandb.init(project="multi-subarrays-doa", job_type="test")

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        batch_size=args.batch_size,
        doa_only=args.train_doa_only,
        profiler=profiler,
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    if args.train_doa_only:
        acc_deg = math.degrees(math.sqrt(test_metrics["rmspe_sq"] / test_metrics["num_sources"]))
        mean_ue = test_metrics["ue_loss"]
        sigma_mae = test_metrics["sigma_mae"]

        print(f"\n{'=' * 50}")
        print("📊 TEST EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"Combined Total Loss      : {test_metrics['total_loss']:.6f}")
        print(f"DOA Accuracy (Avg/Src)   : {acc_deg:.3f}°")
        print(f"Reliability (Mean UE)    : {mean_ue:.3e}")
        print(f"Pred vs True Sigma (MAE) : {sigma_mae:.3f}°  <-- Network Variance Accuracy")
        print(f"{'=' * 50}\n")

        if args.log_to_wandb:
            wandb.log({
                "test_total_loss": test_metrics["total_loss"],
                "test_acc_degrees": acc_deg,
                "test_mean_ue_loss": mean_ue,
                "test_pred_vs_true_sigma_mae": sigma_mae
            })
    else:
        print(f"Test Loss (MSE): {test_metrics['total_loss']:.6f}")
        if args.log_to_wandb:
            wandb.log({"test_loss": test_metrics["total_loss"]})

    if args.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    main()