# ---------------- main.py ----------------
"""
Full trainer revised: now splits the loaded dataset into **train** and **validation** subsets and
reports validation loss every `--val_freq` epochs.  
Sensor positions are still auto‑extracted from the dataset, so no CLI flag is needed.
"""
# Add the project root to Python path if not already there
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

import random
import numpy as np

from src.multi_subarrays_model import MultiSubarraysModel
from src.multi_model_dataset import SensorSourceGraphDataset, Sensor, Source
from src.system_model import SystemModelParams
from src.models import ModelGenerator
from src.criterions import RMSPELoss

# -----------------------------------------------------------------------------
# collate_fn – keeps heterogeneous objects intact
# -----------------------------------------------------------------------------

def graph_scene_collate(batch):
    sensor_positions, source_positions, IQ_signals, doa_stacks = zip(*batch)
    return (
        torch.stack(sensor_positions, dim=0),    # (batch, n_sensors, 2)
        torch.stack(source_positions, dim=0),    # (batch, n_sources, 2)
        torch.stack(IQ_signals, dim=0),          # (batch, n_arrays, N, T, n_samples)
        torch.stack(doa_stacks, dim=0)           # (batch, n_arrays, M)
    )


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- Seed ----------------
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # ---------------- WandB ----------------
        wandb.init(
            project="multi-subarrays-doa",
            config=vars(args),
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # ---------------- Dataset ----------------
        full_ds: SensorSourceGraphDataset = torch.load(args.dataset_path, weights_only=False)
        # Set to True to use pre-computed graph features for faster batching
        # Set to False to use original graphs
        full_ds.set_use_graph_features(use_features=True)

        val_len = int(len(full_ds) * args.val_split)
        train_len = len(full_ds) - val_len
        self.train_ds, self.val_ds = random_split(
            full_ds,
            lengths=[train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed),  # reproducible split
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=graph_scene_collate,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=graph_scene_collate,
            drop_last=True,
        )

        # ---------------- Model ----------------
        self.model = self._build_model().to(self.device)

        # --- Log config artificat to wandb
        configuration_artifact = wandb.Artifact(
            name="config_json",
            type="config",  # or "config", "results", etc.
        )
        configuration_artifact.add_file(self.args.config_path)
        wandb.log_artifact(configuration_artifact)

        # Configure which parts of the model are trainable
        self._configure_trainable_params()

        # ---------------- Optimiser / Scheduler ----------------
        self.optimizer = optim.Adam(
            self.trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
        # TODO: ADD periodic mse loss
        if self.args.train_doa_only:
            self.criterion = RMSPELoss()
        else:
            self.criterion = nn.MSELoss()

        # Resume ------------------------------------------------
        if args.resume:
            self._load_checkpoint(args.resume)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _configure_trainable_params(self):
        """
        If args.train_scopes is None: train everything (original behavior).
        Otherwise:
          - freeze all parameters
          - unfreeze only those whose name contains any of the scopes
        """
        if self.args.train_scopes is None:
            # default: train everything
            self.trainable_params = list(self.model.parameters())
            return

        # Freeze all params
        for p in self.model.parameters():
            p.requires_grad = False

        scopes = self.args.train_scopes
        for name, p in self.model.named_parameters():
            if any(scope in name for scope in scopes):
                p.requires_grad = True

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not self.trainable_params:
            raise ValueError(
                f"No parameters matched train_scopes={scopes}. "
                f"Example: try --train_scopes subarray_models"
            )

        # Optional: print which parts are trainable
        print("Trainable parameter groups:")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print("  ", name)

    def _extract_sensor_positions(self):
        sensor_positions = self.train_ds.dataset.get_sensor_potision()
        if sensor_positions.any():
            return sensor_positions
        else:
            ValueError("Sensors Position Not Found")


    def _build_model(self):
        return MultiSubarraysModel(
            sensors_positions=self._extract_sensor_positions(),
            multi_model_configuration=self.args.config_path,
            args=self.args
        )

    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print(f"[WARN] Could not load optimizer/scheduler state: {e}")
        return ckpt["epoch"]

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        torch.save(ckpt, self.args.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(ckpt, self.args.checkpoint_dir / "best.pth")

    # ---------------------------------------------------------------------
    # Epoch loops
    # ---------------------------------------------------------------------

    def _loop(self, loader, train: bool):
        if train:
            self.model.train()
            batch_size = self.args.batch_size
        else:
            self.model.eval()
            batch_size = self.args.val_batch_size

        running = 0.0
        pos_gt = None
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for sensor_positions, source_positions, samples , doa_gt in tqdm(loader, desc="Val" if not train else "Train", leave=False): 

                # Move tensors to device
                sensor_positions = sensor_positions.to(self.device)
                source_positions = source_positions.to(self.device)
                samples = samples.to(self.device)
                doa_gt = np.radians(doa_gt.to(self.device))

                if train:
                    self.optimizer.zero_grad()

                # TODO: create accumlated loss
                loss = 0

                # gt_pos is now a tensor of dimension [M - Number of sources]. Each entry is the direction
                # of source i from sensor array [subarray_index]
                if self.args.train_doa_only:
                    model_result = self.model(sensor_positions, samples, doa_gt)
                    #doa_pred, pos_pred, dop = model_result["bearings"], model_result["source_estimated_position"], model_result["dop"]
                    doa_pred = model_result["bearings"]
                    #doa_pred = doa_pred.squeeze(dim=-1)
                    #loss = self.criterion(doa_pred, doa_gt)
                    for i in range(doa_gt.shape[1]):
                        loss += self.criterion(doa_pred[:, i, :], doa_gt.unsqueeze(-1)[:, i, :])
                else:
                    doa_pred, pos_pred, dop = self.model(sensor_positions, samples, source_positions)
                    loss = self.criterion(pos_pred, source_positions.squeeze(-2))

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.model.zero_grad()

                running += loss.item()

        return running / (len(loader) * batch_size)

    # expose convenience wrappers
    def train_epoch(self):
        return self._loop(self.train_loader, train=True)

    def val_epoch(self):
        return self._loop(self.val_loader, train=False)

    # ---------------------------------------------------------------------
    # Main train loop
    # ---------------------------------------------------------------------

    def train(self):
        best_val = float("inf")
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch()
            self.scheduler.step()

            wandb.log({"epoch": epoch, "train_loss": train_loss, "lr": self.optimizer.param_groups[0]["lr"]})

            # ---- Validation ----
            if epoch % self.args.val_freq == 0 or epoch == self.args.epochs - 1:
                val_loss = self.val_epoch()
                wandb.log({"epoch": epoch, "val_loss": val_loss})

                if val_loss < best_val:
                    best_val = val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"[Epoch {epoch:03d}] *new best*  train={train_loss:.6f}  val={val_loss:.6f}")
                else:
                    print(f"[Epoch {epoch:03d}]            train={train_loss:.6f}  val={val_loss:.6f}")
            else:
                self._save_checkpoint(epoch, is_best=False)
                print(f"[Epoch {epoch:03d}] train={train_loss:.6f}")

            if self.args.visualize:
                self._visualize_fixed_sample(sample_idx=self.args.visualize_sample_index, epoch=epoch, batch_size=self.args.batch_size)

        wandb.finish()

        # Create the GIF
        if self.args.visualize:
            from src.visualization import create_sample_gif
            create_sample_gif(sample_idx=self.args.visualize_sample_index)

            for index in range(5):
                create_sample_gif(sample_idx=index)


    # ---------------------------------------------------------------------
    def get_fixed_batch(self, batch_size=100, start_index=0):
        # Subset maps logical index → real dataset index
        subset_indices = self.train_ds.indices[start_index: start_index + batch_size]

        # Get raw samples from dataset using real indices
        samples = [self.train_ds.dataset[i] for i in subset_indices]  # list of tuples

        # Collate using the same logic as DataLoader
        batch = graph_scene_collate(samples)

        # Move to device
        batch = tuple(x.to(self.device) if torch.is_tensor(x) else x for x in batch)

        return batch

    def _visualize_fixed_sample(self, sample_idx: int, epoch: int, batch_size):
        from src.visualization import visualize_ray_frame

        # Get the specific item directly from the full dataset (not the split subset!)
        batch = self.get_fixed_batch(batch_size=batch_size, start_index=0)
        sensor_pos, source_pos, iq_signal, doa_gt = batch

        sensor_pos = sensor_pos.to(self.device)  # (1, M, 2)
        source_pos = source_pos.to(self.device)  # (1, 1, 2)
        iq_signal = iq_signal.to(self.device)  # (1, ...)
        doa_gt = doa_gt.to(self.device)

        with torch.no_grad():
            model_result = self.model(sensor_pos, iq_signal, doa_gt)
            #doa_pred, pos_pred = model_result["bearings"], model_result["source_estimated_position"]
            doa_pred = model_result["bearings"]

        for sample_index in range(5):
            visualize_ray_frame(
                positions=sensor_pos[sample_index],  # (M, 2)
                bearings=doa_pred[sample_index],  # (M,)
                #x_hat=pos_pred[sample_index],  # (2,)
                x_hat=torch.zeros_like(source_pos[sample_index]),
                x_true=source_pos[sample_index],  # (2,)
                step=epoch,
                save_path=f"visualizations/sample_{sample_index:03d}_epoch_{epoch:03d}.png"
            )

            import wandb
            #wandb.log({"epoch": epoch, "doa_pred": doa_pred[sample_index], "pos_pred": pos_pred[sample_index], "doa_gt": doa_gt[sample_index], "pos_gt": source_pos[sample_index, 0]})
            wandb.log({"epoch": epoch, "doa_pred": doa_pred[sample_index],
                       "doa_gt": doa_gt[sample_index], "pos_gt": source_pos[sample_index]})
        if self.args.visualize and self.args.log_to_wandb:

            wandb.log({
                f"viz/sample_{sample_idx}/epoch_{epoch:03d}":
                    wandb.Image(f"visualizations/sample_{sample_idx:03d}_epoch_{epoch:03d}.png")
            })


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Train Multi‑Subarrays Model")

    # Data
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--config_path", required=True)
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data held out for validation")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training hyper‑params
    p.add_argument("--train_doa_only", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--val_batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_step_size", type=int, default=20)
    p.add_argument("--lr_gamma", type=float, default=0.2)
    p.add_argument("--val_freq", type=int, default=1, help="Validate every N epochs")
    p.add_argument(
        "--train_scopes",
        nargs="+",
        default=None,
        help=(
            "List of substrings of parameter names to train; others are frozen. "
            "Example: --train_scopes subarray_models learned_attentaion"
        ),
    )

    # System / misc
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)

    #Visualization
    p.add_argument("--visualize", action="store_true", help="Enable per-epoch sample visualization")
    p.add_argument("--visualize_sample_index", type=int, default=0, help="Index of the sample to track")
    p.add_argument("--log_to_wandb", action="store_true", help="Log visualizations to W&B")

    return p.parse_args()


def main():
    args = parse_args()
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    Trainer(args).train()


if __name__ == "__main__":
    main()
