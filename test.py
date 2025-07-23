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


@torch.no_grad()
def evaluate(model, loader, criterion, device, batch_size, doa_only):
    model.eval()
    total_loss = 0.0
    for sensor_positions, source_positions, samples, doa_gt in loader:
        sensor_positions = sensor_positions.to(device)
        source_positions = source_positions.to(device)
        samples = samples.to(device)
        doa_gt = doa_gt.to(device)

        if doa_only:
            pred = model(sensor_positions, samples, doa_gt)
            loss = criterion(pred, doa_gt)
        else:
            pred = model(sensor_positions, samples, None)
            loss = criterion(pred, source_positions)

        total_loss += loss.item()

    return total_loss / (len(loader) * batch_size)


def main():
    parser = argparse.ArgumentParser("Test Multi‑Subarrays Model")
    parser.add_argument("--test_dataset_path", required=True, help="Path to test dataset .pt file")
    parser.add_argument("--config_path", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--train_doa_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=100)
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

    # Load best checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    match_learned_attn_shapes(model, checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model_state_dict"])

    # Criterion
    criterion = RMSPELoss() if args.train_doa_only else torch.nn.MSELoss()

    # Optional W&B logging
    if args.log_to_wandb:
        wandb.init(project="multi-subarrays-doa", job_type="test")

    # Evaluate
    test_loss = evaluate(model, test_loader, criterion, device, batch_size=args.batch_size, doa_only=args.train_doa_only)
    print(f"Test loss: {test_loss:.6f}")

    if args.log_to_wandb:
        wandb.log({"test_loss": test_loss})
        wandb.finish()


if __name__ == "__main__":
    main()
