import sys
import os
import math
import itertools
from signal import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
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
    Calculates the 'true' analytical uncertainty (sigma in degrees) using the covariance matrix Rx.
    """
    device = doa_true_rad.device
    iq_samples = samples.squeeze(dim=2)
    signal_shape_uncertainty = samples.shape[-1]

    uncertainty_block = UncertaintyEstimation(signal_shape=signal_shape_uncertainty).to(device)
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
                title=f"Sigma (with True Rx) vs True DoA (Subarray {index})"
            )

    return sigma_true


def get_theoretical_ue(doa_pred, doa_gt, sigma_true_rad):
    """
    Calculates the exact UE metric using the True Theoretical Sigma.
    """
    device = doa_pred.device
    B, S, P = doa_pred.shape

    perm_indices = torch.tensor(list(itertools.permutations(range(P))), device=device)
    num_perms = perm_indices.shape[0]

    preds_exp = doa_pred.unsqueeze(2).expand(B, S, num_perms, P)
    perms_exp = perm_indices.view(1, 1, num_perms, P).expand(B, S, num_perms, P)
    preds_perm = torch.gather(preds_exp, dim=3, index=perms_exp)

    targets_exp = doa_gt.unsqueeze(2)
    error = (((preds_perm - targets_exp) + (np.pi / 2)) % np.pi) - np.pi / 2

    rmspe_squared_all = torch.mean(error ** 2, dim=3)
    best_perm_idx = torch.argmin(rmspe_squared_all, dim=2, keepdim=True)
    best_perm_idx_exp = best_perm_idx.unsqueeze(3).expand(B, S, 1, P)

    best_error = torch.gather(error, dim=2, index=best_perm_idx_exp).squeeze(2)

    empirical_variance = best_error ** 2
    true_variance = sigma_true_rad ** 2

    ue_loss_per_source = (true_variance - empirical_variance) ** 2
    accumulated_ue = torch.sum(ue_loss_per_source, dim=2)
    return torch.mean(accumulated_ue).item()


@torch.no_grad()
def evaluate(model, loader, criterion, device, batch_size, doa_only, profiler=None, run_esprit=False):
    model.eval()

    total_total_loss = 0.0
    total_rmspe_sq = 0.0
    total_net_ue_loss = 0.0
    total_true_ue_loss = 0.0
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

                    if run_esprit:
                        from src.models import esprit

                        iq_samples = samples.squeeze(dim=2)
                        Rx_batch_list = []

                        for b_idx in range(iq_samples.shape[0]):
                            Rx = []
                            for subarray_index in range(iq_samples.shape[1]):
                                Rx.append(torch.cov(iq_samples[b_idx, subarray_index, :, :]))
                            Rx_batch_list.append(torch.stack(Rx))
                        RX_batch = torch.stack(Rx_batch_list).to(device)

                        doa_preds = []
                        for subarray_index in range(iq_samples.shape[1]):
                            subarray_Rx = RX_batch[:, subarray_index, :, :]
                            subarray_doa, _ = esprit(subarray_Rx, num_sources, subarray_Rx.shape[0])
                            doa_preds.append(subarray_doa)

                        doa_pred = torch.stack(doa_preds, dim=1).to(device)
                        sigma_pred_deg = calculate_true_uncertainty(doa_gt, samples, plot=False)
                        pos_pred = torch.zeros_like(source_positions)
                    else:
                        model_result = model(sensor_positions, samples, doa_gt)
                        doa_pred = model_result["bearings"]
                        sigma_pred_deg = model_result["sigma_i"]
                        if model.estimate_position:
                            pos_pred = model_result.get("source_estimated_position", torch.zeros_like(source_positions))
                        else:
                            pos_pred = torch.zeros_like(source_positions)

                    sigma_true_deg = calculate_true_uncertainty(doa_gt, samples, plot=False)

                    # Dynamic Loss function calculates values
                    sigma_pred_rad = torch.deg2rad(sigma_pred_deg)
                    loss_out = criterion(sigma_pred_rad, doa_pred, doa_gt)

                    # Handle varying outputs based on the criterion used
                    if isinstance(loss_out, tuple) and len(loss_out) == 3:
                        loss, rmspe_sq, net_ue_loss = loss_out
                    elif isinstance(loss_out, tuple) and len(loss_out) == 2:
                        loss, rmspe_sq = loss_out
                        net_ue_loss = 0.0
                    else:
                        loss = loss_out
                        rmspe_sq = loss_out  # Fallback if just MSE
                        net_ue_loss = 0.0

                    sigma_true_rad = torch.deg2rad(sigma_true_deg)
                    true_ue_loss = get_theoretical_ue(doa_pred, doa_gt, sigma_true_rad)

                    total_total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                    total_rmspe_sq += rmspe_sq.item() if isinstance(rmspe_sq, torch.Tensor) else rmspe_sq
                    total_net_ue_loss += net_ue_loss.item() if isinstance(net_ue_loss, torch.Tensor) else net_ue_loss
                    total_true_ue_loss += true_ue_loss

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
        "net_ue_loss": total_net_ue_loss / divisor,
        "true_ue_loss": total_true_ue_loss / divisor,
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

    # Validation/ESPRIT Arguments
    parser.add_argument("--esprit_baseline", action="store_true", help="Run ESPRIT instead of the Neural Network")
    parser.add_argument("--visualize", action="store_true", help="Generate frame visualizations for the test set")
    parser.add_argument("--visualize_num", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--visualize_sigma", action="store_true", help="Plot Predicted and True Sigma vs DoA")
    parser.add_argument("--viz_prefix", type=str, default="test", help="Subfolder name for visualizations")

    # --- NEW: Dynamic Criterion Parsing ---
    parser.add_argument("--loss_function", type=str, default="CombinedUncertaintyLoss",
                        help="Loss Function to test with")
    parser.add_argument("--lambda_val", type=float, default=0.5, help="Lambda value for UE Loss")
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

    # Only load NN weights if we are NOT running ESPRIT
    if not args.esprit_baseline:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        match_learned_attn_shapes(model, checkpoint["model_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"])

    # --- NEW: Instantiate Correct Criterion dynamically! ---
    if args.train_doa_only:
        if args.loss_function == "CombinedUncertaintyLoss":
            criterion = CombinedUncertaintyLoss(lambda_val=args.lambda_val, reduction='mean')
        elif args.loss_function == "RMSPELoss":
            criterion = RMSPELoss()
        elif args.loss_function == "MSELoss":
            criterion = torch.nn.MSELoss()
        else:
            criterion = CombinedUncertaintyLoss(lambda_val=args.lambda_val, reduction='mean')  # Safe fallback
    else:
        criterion = torch.nn.MSELoss()

    if args.log_to_wandb and not args.esprit_baseline:
        wandb.init(project="multi-subarrays-doa", job_type="test")

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        batch_size=args.batch_size,
        doa_only=args.train_doa_only,
        profiler=profiler,
        run_esprit=args.esprit_baseline
    )

    # ---------------------------------------------------------
    # PRINT RESULTS
    # ---------------------------------------------------------
    if args.train_doa_only:
        acc_deg = math.degrees(math.sqrt(test_metrics["rmspe_sq"] / test_metrics["num_sources"]))
        title = "ESPRIT BASELINE RESULTS" if args.esprit_baseline else "TEST EVALUATION RESULTS"

        print(f"\n{'=' * 55}")
        print(f"📊 {title} (Loss: {args.loss_function})")
        print(f"{'=' * 55}")
        print(f"Combined Total Loss      : {test_metrics['total_loss']:.6f}")
        print(f"DOA Accuracy (Avg/Src)   : {acc_deg:.3f}°\n")

        print(f"Reliability (Accumulated UE Loss):")
        print(f"  - Network/ESPRIT UE Loss: {test_metrics['net_ue_loss']:.3e}")
        print(f"  - Theoretical UE Loss  : {test_metrics['true_ue_loss']:.3e}")
        print(f"{'=' * 55}\n")

        if args.log_to_wandb and not args.esprit_baseline:
            wandb.log({
                "test_total_loss": test_metrics["total_loss"],
                "test_acc_degrees": acc_deg,
                "test_network_ue_loss": test_metrics["net_ue_loss"],
                "test_theoretical_ue_loss": test_metrics["true_ue_loss"],
            })

    # ---------------------------------------------------------
    # VISUALIZATION LOGIC
    # ---------------------------------------------------------
    if args.visualize or args.visualize_sigma:
        print(f"🎨 Generating Visualizations for {args.viz_prefix}...")

        batch = next(iter(test_loader))
        sensor_pos, source_pos, iq_samples, doa_gt = batch

        sensor_pos = sensor_pos.to(device)
        source_pos = source_pos.to(device)
        iq_samples = iq_samples.to(device)
        doa_gt = torch.deg2rad(doa_gt).to(device)

        model.eval()
        with torch.no_grad():
            if args.train_doa_only:
                if args.esprit_baseline:
                    from src.models import esprit
                    iq_sq = iq_samples.squeeze(dim=2)
                    Rx_batch_list = []
                    for b_idx in range(iq_sq.shape[0]):
                        Rx = []
                        for subarray_index in range(iq_sq.shape[1]):
                            Rx.append(torch.cov(iq_sq[b_idx, subarray_index, :, :]))
                        Rx_batch_list.append(torch.stack(Rx))
                    RX_batch = torch.stack(Rx_batch_list).to(device)

                    doa_preds = []
                    for subarray_index in range(iq_sq.shape[1]):
                        subarray_Rx = RX_batch[:, subarray_index, :, :]
                        subarray_doa, _ = esprit(subarray_Rx, doa_gt.shape[-1], subarray_Rx.shape[0])
                        doa_preds.append(subarray_doa)

                    doa_pred = torch.stack(doa_preds, dim=1).to(device)
                    sigma_pred_deg = calculate_true_uncertainty(doa_gt, iq_samples, plot=False)
                    pos_pred = torch.zeros_like(source_pos)
                else:
                    model_result = model(sensor_pos, iq_samples, doa_gt)
                    doa_pred = model_result["bearings"]
                    sigma_pred_deg = model_result["sigma_i"]
                    pos_pred = model_result.get("source_estimated_position", torch.zeros_like(source_pos))
            else:
                pos_pred = model(sensor_pos, iq_samples, None)
                doa_pred = torch.zeros(iq_samples.shape[0], iq_samples.shape[1], source_pos.shape[1]).to(device)
                sigma_pred_deg = None

        # --- Plot Position / DoA Frames ---
        if args.visualize:
            viz_dir = f"visualizations/{args.viz_prefix}"
            os.makedirs(viz_dir, exist_ok=True)
            num_vis = min(args.visualize_num, iq_samples.shape[0])
            for sample_index in range(num_vis):
                save_path = f"{viz_dir}/sample_{sample_index:03d}.png"

                current_sigmas = None
                if sigma_pred_deg is not None:
                    current_sigmas = torch.deg2rad(sigma_pred_deg[sample_index])

                visualize_ray_frame(
                    positions=sensor_pos[sample_index],
                    bearings=doa_pred[sample_index],
                    x_hat=pos_pred[sample_index],
                    x_true=source_pos[sample_index],
                    sigmas=current_sigmas,
                    step=0,
                    save_path=save_path
                )

            print(f"✅ Saved {num_vis} frame visualizations to '{viz_dir}/'")

        # --- Plot Sigma vs DoA ---
        if args.visualize_sigma and args.train_doa_only:
            print("📈 Generating Sigma vs DoA plots...")
            sigma_true_deg = calculate_true_uncertainty(doa_gt, iq_samples, plot=False)

            for subarray_index in range(doa_gt.shape[1]):
                plot_sigma_vs_doa(
                    torch.rad2deg(doa_gt[:, subarray_index, :]).cpu(),
                    sigma_pred_deg[:, subarray_index, :].cpu(),
                    title=f"[{args.viz_prefix}] Predicted Sigma vs True DoA (Subarray {subarray_index})"
                )
                plot_sigma_vs_doa(
                    torch.rad2deg(doa_gt[:, subarray_index, :]).cpu(),
                    sigma_true_deg[:, subarray_index, :].cpu(),
                    title=f"[{args.viz_prefix}] True Sigma vs True DoA (Subarray {subarray_index})"
                )
            print("✅ Sigma plots generated.")

    if args.log_to_wandb and not args.esprit_baseline:
        wandb.finish()


if __name__ == "__main__":
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    main()