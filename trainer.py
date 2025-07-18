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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=graph_scene_collate,
            drop_last=True,
        )

        # ---------------- Model ----------------
        self.model = self._build_model().to(self.device)

        # ---------------- Optimiser / Scheduler ----------------
        self.optimizer = optim.Adam(
            self.model.parameters(),
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
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
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
        else:
            self.model.eval()

        running = 0.0
        pos_gt = None
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for sensor_positions, source_positions, samples , doa_gt in tqdm(loader, desc="Val" if not train else "Train", leave=False): 

                # Move tensors to device
                sensor_positions = sensor_positions.to(self.device)
                source_positions = source_positions.to(self.device)
                samples = samples.to(self.device)
                doa_gt = doa_gt.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                # gt_pos is now a tensor of dimension [M - Number of sources]. Each entry is the direction
                # of source i from sensor array [subarray_index]
                if self.args.train_doa_only:
                    doa_pred = self.model(sensor_positions, samples, doa_gt)
                    loss = self.criterion(doa_pred, doa_gt)
                else:
                    pos_pred = self.model(sensor_positions, samples, pos_gt)
                    loss = self.criterion(pos_pred, pos_gt)

                if train:
                    loss.backward()
                    self.optimizer.step()

                running += loss.item()

        return running / (len(loader) * self.args.batch_size)

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

        wandb.finish()


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
    p.add_argument("--val_batch_size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_step_size", type=int, default=20)
    p.add_argument("--lr_gamma", type=float, default=0.2)
    p.add_argument("--val_freq", type=int, default=5, help="Validate every N epochs")

    # System / misc
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    Trainer(args).train()


if __name__ == "__main__":
    main()