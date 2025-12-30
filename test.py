import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
from torch.utils.data import DataLoader
import wandb

from trainer import graph_scene_collate
from src.multi_subarrays_model import MultiSubarraysModel
from src.multi_model_dataset import SensorSourceGraphDataset
from src.criterions import RMSPELoss

# To match attenation layers
from src.learned_agg_layer import match_learned_attn_shapes
from src.localization_block import position_errors

from src.visualization import visualize_ray_frame


from torch.profiler import (
    profile,
    schedule,
    tensorboard_trace_handler,
    ProfilerActivity,
    record_function,
)


@torch.no_grad()
def evaluate(model, loader, criterion, device, batch_size, doa_only, profiler=None):
    model.eval()
    total_loss = 0.0
    loss = 0.0
    for step, (sensor_positions, source_positions, samples, doa_gt) in enumerate(loader):
        with record_function("eval_step"):
            sensor_positions = sensor_positions.to(device)
            source_positions = source_positions.to(device)
            samples = samples.to(device)
            doa_gt = torch.deg2rad(doa_gt).to(device)

            if doa_only:
                with record_function("model_forward_doa"):
                    model_result = model(sensor_positions, samples, doa_gt)


                    #doa_pred, pos_pred, dop = model_result["bearings"], model_result["source_estimated_position"], model_result["dop"]
                    doa_pred = model_result["bearings"]
                    sigma_pred = model_result["sigma_i"]

                    if model.estimate_position is True:
                        pos_pred = model_result["source_estimated_position"]
                        if model.estimate_uncertainty is True:
                            position_metrics = position_errors(model_result["source_estimated_position"], model_result["source_estimated_position_wls"],
                                                           source_positions)
                            model_result["position_metrics"] = position_metrics

                    else:
                        pos_pred = torch.zeros_like(source_positions[sample_index])


                    for i in range(doa_gt.shape[1]):
                        loss += criterion(doa_pred[:, i, :], doa_gt[:, i, :])

                    for sample_index in range(10):
                        visualize_ray_frame(
                            positions=sensor_positions[sample_index],  # (M, 2)
                            bearings=doa_pred[sample_index],  # (M,)
                            x_hat=pos_pred[sample_index],  # (2,)
                            x_true=source_positions[sample_index],  # (2,)
                            sigmas=sigma_pred[sample_index],
                            #sigmas=torch.zeros_like(doa_pred[sample_index]),
                            step=step,
                            save_path=f"visualizations/sample_{sample_index:03d}_test_{step:03d}.png"
                        )



            else:
                with record_function("model_forward_pos"):
                    pred = model(sensor_positions, samples, None)
                loss = criterion(pred, source_positions)

            total_loss += loss.item()

        # advance profiler step at the end of each iteration
        if profiler is not None:
            profiler.step()

    return total_loss / (len(loader) * batch_size)


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

    # Load test dataset
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

    # Extract sensor positions from test dataset
    sensor_positions = test_ds.get_sensor_potision()

    # Build model
    model = MultiSubarraysModel(
        sensors_positions=sensor_positions,
        multi_model_configuration=args.config_path,
        args=args
    ).to(device)

    # Estimate the uncertainty
    model.estimate_uncertainty = True
    model.estimate_position = True

    # Load best checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    match_learned_attn_shapes(model, checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    # Criterion
    criterion = RMSPELoss() if args.train_doa_only else torch.nn.MSELoss()

    # Optional W&B logging
    if args.log_to_wandb:
        wandb.init(project="multi-subarrays-doa", job_type="test")

    # Evaluate (optionally under profiler)
    test_loss = evaluate(
        model,
        test_loader,
        criterion,
        device,
        batch_size=args.batch_size,
        doa_only=args.train_doa_only,
        profiler=profiler,
    )
    print(f"Test loss: {test_loss:.6f}")

    if args.log_to_wandb:
        wandb.log({"test_loss": test_loss})
        wandb.finish()


if __name__ == "__main__":
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    main()