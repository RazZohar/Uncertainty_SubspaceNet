# ---------------- main.py ----------------
"""
Full trainer revised: now splits the loaded dataset into **train** and **validation** subsets and
reports validation loss every `--val_freq` epochs.  
Sensor positions are still auto‑extracted from the dataset, so no CLI flag is needed.
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from multi_subarrays_model import MultiSubarraysModel
from multi_model_dataset import SensorSourceGraphDataset, Sensor, Source
from system_model import SystemModelParams
from models import ModelGenerator

# -----------------------------------------------------------------------------
# collate_fn – keeps heterogeneous objects intact
# -----------------------------------------------------------------------------

def graph_scene_collate(batch):
    """batch: List[(graph, samples, gt_pos_tensor)]"""
    graphs, samples, gt_pos = zip(*batch)  # tuples of length B
    return list(graphs), list(samples), torch.stack(gt_pos, dim=0)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------- WandB ----------------
        wandb.init(
            project="multi-subarrays-doa",
            config=vars(args),
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # ---------------- Dataset ----------------
        full_ds: SensorSourceGraphDataset = torch.load(args.dataset_path, weights_only=False)

        val_len = int(len(full_ds) * args.val_split)
        train_len = len(full_ds) - val_len
        self.train_ds, self.val_ds = random_split(
            full_ds,
            lengths=[train_len, val_len],
            generator=torch.Generator().manual_seed(42),  # reproducible split
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=graph_scene_collate,
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=graph_scene_collate,
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
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for model_graphs, samples in tqdm(loader, desc="Val" if not train else "Train", leave=False):
                
                # gt_pos

                if train:
                    self.optimizer.zero_grad()

                pred_pos = self.model((model_graphs, samples))

                loss = self.criterion(pred_pos, gt_pos)

                if train:
                    loss.backward()
                    self.optimizer.step()

                running += loss.item()

        return running / len(loader)

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

    # Training hyper‑params
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_step_size", type=int, default=30)
    p.add_argument("--lr_gamma", type=float, default=0.1)
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