# ---------------- main.py ----------------
"""
Full trainer revised: Supports multi-stage training curricula via a JSON configuration.
Properly handles dynamic parameter freezing/unfreezing and tracks continuous
global epochs for accurate Weights & Biases logging.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

import random
import numpy as np
import math

from src.multi_subarrays_model import MultiSubarraysModel
from src.multi_model_dataset import SensorSourceGraphDataset, Sensor, Source
from src.system_model import SystemModelParams
from src.models import ModelGenerator
from src.criterions import RMSPELoss

# If you implemented UEELoss, you can import it here
from src.criterions import UEELoss, CombinedUncertaintyLoss


# -----------------------------------------------------------------------------
# collate_fn – keeps heterogeneous objects intact
# -----------------------------------------------------------------------------

def graph_scene_collate(batch):
    sensor_positions, source_positions, IQ_signals, doa_stacks = zip(*batch)
    return (
        torch.stack(sensor_positions, dim=0),  # (batch, n_sensors, 2)
        torch.stack(source_positions, dim=0),  # (batch, n_sources, 2)
        torch.stack(IQ_signals, dim=0),  # (batch, n_arrays, N, T, n_samples)
        torch.stack(doa_stacks, dim=0)  # (batch, n_arrays, M)
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
        full_ds.set_use_graph_features(use_features=True)

        val_len = int(len(full_ds) * args.val_split)
        train_len = len(full_ds) - val_len

        # Save datasets as class variables so DataLoaders can be dynamically rebuilt
        self.train_ds, self.val_ds = random_split(
            full_ds,
            lengths=[train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed),
        )

        # ---------------- Model ----------------
        self.model = self._build_model().to(self.device)

        self.model.estimate_uncertainty = True

        # --- Log config artifacts to wandb ---
        configuration_artifact = wandb.Artifact(name="config_json", type="config")
        configuration_artifact.add_file(self.args.config_path)
        if hasattr(self.args, 'stages_config') and self.args.stages_config:
            configuration_artifact.add_file(self.args.stages_config)
        wandb.log_artifact(configuration_artifact)

        # Initialize current stage variables to default args
        self.current_doa_only = args.train_doa_only
        self.current_train_scopes = args.train_scopes
        self.start_global_epoch = 0

        # Resume ------------------------------------------------
        if args.resume:
            self.start_global_epoch = self._load_checkpoint(args.resume)

    # ---------------------------------------------------------------------
    # Multi-Stage Helpers
    # ---------------------------------------------------------------------

    def _setup_stage(self, stage_config, stage_idx):
        print(f"\n{'=' * 60}")
        print(f"🚀 INITIALIZING STAGE {stage_idx + 1}: {stage_config.get('name', 'Unnamed Stage')}")
        print(f"{'=' * 60}")

        # 1. Update State Variables
        self.current_epochs = stage_config.get("epochs", self.args.epochs)
        self.current_doa_only = stage_config.get("train_doa_only", self.args.train_doa_only)
        self.current_train_scopes = stage_config.get("train_scopes", self.args.train_scopes)

        current_batch_size = stage_config.get("batch_size", self.args.batch_size)
        current_lr = stage_config.get("learning_rate", self.args.learning_rate)

        # 2. Re-configure frozen/unfrozen parameters
        self._configure_trainable_params()

        # 3. Re-initialize Optimizer with the correctly unfrozen parameters
        self.optimizer = optim.Adam(
            self.trainable_params,
            lr=current_lr,
            weight_decay=self.args.weight_decay,
        )

        # 4. Re-initialize Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=stage_config.get("lr_step_size", self.args.lr_step_size),
            gamma=stage_config.get("lr_gamma", self.args.lr_gamma)
        )

        # 5. Re-initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_ds, batch_size=current_batch_size, shuffle=True,
            num_workers=self.args.num_workers, collate_fn=graph_scene_collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.args.val_batch_size, shuffle=False,
            num_workers=self.args.num_workers, collate_fn=graph_scene_collate, drop_last=True,
        )

        # 6. Re-initialize Loss Function
        loss_name = stage_config.get("loss_function", "MSELoss")

        if loss_name == "CombinedUncertaintyLoss":
            # Extract lambda from the JSON, default to 0.75 if not found
            stage_lambda = stage_config.get("lambda_val", 0.75)
            self.criterion = CombinedUncertaintyLoss(lambda_val=stage_lambda)
            print(f"   * Loss: {loss_name} (Lambda: {stage_lambda})")

        elif loss_name == "RMSPELoss":
            self.criterion = RMSPELoss()
            print(f"   * Loss: {loss_name}")

        elif loss_name == "MSELoss":
            self.criterion = nn.MSELoss()
            print(f"   * Loss: {loss_name}")

        else:
            raise ValueError(f"Unknown loss function requested: {loss_name}")

        print(f"   * Epochs: {self.current_epochs}")
        print(f"   * LR: {current_lr} | Batch Size: {current_batch_size}")
        print(f"   * Loss: {loss_name} | DOA Only: {self.current_doa_only}\n")

    def _configure_trainable_params(self):
        """
        Dynamically freezes/unfreezes model parameters based on self.current_train_scopes.
        """
        # Freeze everything first to ensure a clean slate
        for p in self.model.parameters():
            p.requires_grad = False

        scopes = self.current_train_scopes

        if scopes is None:
            # If no scopes provided, unfreeze everything
            for p in self.model.parameters():
                p.requires_grad = True
            print("   * Trainable Scopes: ALL (Full Model Unfrozen)")
        else:
            # Unfreeze only the requested scopes
            for name, p in self.model.named_parameters():
                if any(scope in name for scope in scopes):
                    p.requires_grad = True
            print(f"   * Trainable Scopes: {scopes}")

        # Build the final list of parameters that the optimizer should track
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        if not self.trainable_params:
            raise ValueError(f"No parameters matched train_scopes={scopes}.")

    def _extract_sensor_positions(self):
        sensor_positions = self.train_ds.dataset.get_sensor_potision()
        if sensor_positions.any():
            return sensor_positions
        else:
            raise ValueError("Sensors Position Not Found")

    def _build_model(self):
        return MultiSubarraysModel(
            sensors_positions=self._extract_sensor_positions(),
            multi_model_configuration=self.args.config_path,
            args=self.args
        )

    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state_dict"])
        # We only return the epoch. The optimizer/scheduler states are intentionally
        # ignored here because multi-stage dynamically rebuilds them.
        return ckpt.get("global_epoch", ckpt.get("epoch", 0))

    def _save_checkpoint(self, global_epoch: int, is_best: bool = False, stage_name: str = None,
                         stage_prefix: str = None):
        ckpt = {
            "global_epoch": global_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        # Always save latest
        torch.save(ckpt, self.args.checkpoint_dir / "latest.pth")

        # --- Save best for specific stage using prefix ---
        if is_best:
            best_filename = f"best_{stage_prefix}.pth" if stage_prefix else "best.pth"
            torch.save(ckpt, self.args.checkpoint_dir / best_filename)

        # Save the stage completion snapshot
        if stage_name:
            torch.save(ckpt, self.args.checkpoint_dir / f"{stage_name}.pth")

    # ---------------------------------------------------------------------
    # Epoch loops
    # ---------------------------------------------------------------------

    def _loop(self, loader, train: bool):
        if train:
            self.model.train()
            batch_size = loader.batch_size
        else:
            self.model.eval()
            batch_size = loader.batch_size

        # Track all three metrics
        running_total_loss = 0.0
        running_rmspe_sq = 0.0
        running_ue_loss = 0.0
        current_num_sources = 1  # Dynamically track sources for metric printouts

        ctx = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for sensor_positions, source_positions, samples, doa_gt in tqdm(loader, desc="Train" if train else "Val",
                                                                            leave=False):

                sensor_positions = sensor_positions.to(self.device)
                source_positions = source_positions.to(self.device)
                samples = samples.to(self.device)
                doa_gt = torch.deg2rad(doa_gt).to(self.device)

                # Extract number of sources dynamically
                current_num_sources = doa_gt.shape[-1]

                if train:
                    self.optimizer.zero_grad()

                if self.current_doa_only:
                    model_result = self.model(sensor_positions, samples, doa_gt)
                    doa_pred = model_result["bearings"]

                    # 1. Extract uncertainty (in degrees)
                    uncert_pred_deg = model_result["sigma_i"]

                    # 2. MATCH UNITS: Convert to radians so the loss math works!
                    uncert_pred_rad = torch.deg2rad(uncert_pred_deg)

                    # 3. Calculate combined loss and individual metrics in pure radians
                    loss, rmspe_sq, ue_loss = self.criterion(uncert_pred_rad, doa_pred, doa_gt)

                    # Accumulate for logging
                    running_rmspe_sq += rmspe_sq.item()
                    running_ue_loss += ue_loss.item()
                else:
                    doa_pred, pos_pred, dop = self.model(sensor_positions, samples, source_positions)
                    loss = self.criterion(pos_pred, source_positions.squeeze(-2))

                    # For non-DOA training, we just log zeros for these specific metrics
                    running_rmspe_sq += 0
                    running_ue_loss += 0

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.model.zero_grad()

                running_total_loss += loss.item()

        divisor = len(loader) * batch_size

        # Return a dictionary containing the averaged metrics for this epoch
        return {
            "total_loss": running_total_loss / divisor,
            "rmspe_sq": running_rmspe_sq / divisor,
            "ue_loss": running_ue_loss / divisor,
            "num_sources": current_num_sources
        }

    def train_epoch(self):
        return self._loop(self.train_loader, train=True)

    def val_epoch(self):
        return self._loop(self.val_loader, train=False)

    # ---------------------------------------------------------------------
    # Main train loop
    # ---------------------------------------------------------------------

    def train(self):
        # Load stages from JSON, or fallback to CLI args if not provided
        if hasattr(self.args, 'stages_config') and self.args.stages_config:
            with open(self.args.stages_config, 'r') as f:
                stages = json.load(f).get("stages", [])
        else:
            stages = [{
                "name": "Default Stage (CLI Args)",
                "epochs": self.args.epochs,
                "learning_rate": self.args.learning_rate,
                "train_scopes": self.args.train_scopes,
                "train_doa_only": self.args.train_doa_only,
                "batch_size": self.args.batch_size,
            }]

        # Start global epoch counter (accounts for resuming)
        global_epoch = self.start_global_epoch

        for stage_idx, stage_config in enumerate(stages):

            # --- Reset best validation tracker for THIS stage ---
            best_val = float("inf")

            # --- Extract stage_prefix from JSON (Fallback to 'stage_1', 'stage_2', etc.) ---
            current_stage_prefix = stage_config.get("stage_prefix", f"stage_{stage_idx + 1}")

            # Setup network, optimizer, and loaders for this stage
            self._setup_stage(stage_config, stage_idx)

            # Local epoch tracks progress *within* the current stage
            for local_epoch in range(self.current_epochs):

                # train_epoch now returns a dictionary
                train_metrics = self.train_epoch()
                self.scheduler.step()

                # Log independent train metrics
                wandb.log({
                    "global_epoch": global_epoch,
                    "stage": stage_idx + 1,
                    "stage_local_epoch": local_epoch,
                    "train_total_loss": train_metrics["total_loss"],
                    "train_rmspe_sq": train_metrics["rmspe_sq"],
                    "train_ue_loss": train_metrics["ue_loss"],
                    "lr": self.optimizer.param_groups[0]["lr"]
                })

                # ---- Validation ----
                if global_epoch % self.args.val_freq == 0 or local_epoch == self.current_epochs - 1:

                    val_metrics = self.val_epoch()
                    val_total_loss = val_metrics["total_loss"]

                    # Log independent val metrics to W&B
                    wandb.log({
                        "global_epoch": global_epoch,
                        "val_total_loss": val_total_loss,
                        "val_rmspe_sq": val_metrics["rmspe_sq"],
                        "val_ue_loss": val_metrics["ue_loss"],
                    })

                    # --- CONVERT TO DEGREES FOR PRINTING (USING NUM_SOURCES) ---
                    p_train = train_metrics["num_sources"]
                    p_val = val_metrics["num_sources"]

                    train_acc_deg = math.degrees(math.sqrt(train_metrics["rmspe_sq"] / p_train))
                    val_acc_deg = math.degrees(math.sqrt(val_metrics["rmspe_sq"] / p_val))

                    # Format strings for clean command line output
                    train_str = f"Train Loss: {train_metrics['total_loss']:.5f} (Acc: {train_acc_deg:.3f}°, UE: {train_metrics['ue_loss']:.4e})"
                    val_str = f"Val Loss: {val_total_loss:.5f} (Acc: {val_acc_deg:.3f}°, UE: {val_metrics['ue_loss']:.4e})"

                    if val_total_loss < best_val:
                        best_val = val_total_loss

                        # Save best using the dynamic stage prefix
                        self._save_checkpoint(global_epoch, is_best=True, stage_prefix=current_stage_prefix)
                        print(
                            f"[Stage {stage_idx + 1} | Global Epoch {global_epoch:03d}] *NEW BEST for {current_stage_prefix}* | {train_str} | {val_str}")
                    else:
                        self._save_checkpoint(global_epoch, is_best=False, stage_prefix=current_stage_prefix)
                        print(f"[Stage {stage_idx + 1} | Global Epoch {global_epoch:03d}] {train_str} | {val_str}")
                else:
                    self._save_checkpoint(global_epoch, is_best=False, stage_prefix=current_stage_prefix)

                    # --- CONVERT TO DEGREES FOR PRINTING (TRAIN ONLY) ---
                    p_train = train_metrics["num_sources"]
                    train_acc_deg = math.degrees(math.sqrt(train_metrics["rmspe_sq"] / p_train))
                    train_str = f"Train Loss: {train_metrics['total_loss']:.4f} (Acc: {train_acc_deg:.2f}°, UE: {train_metrics['ue_loss']:.1e})"

                    print(f"[Stage {stage_idx + 1} | Global Epoch {global_epoch:03d}] {train_str}")

                # ---- Visualization ----
                if self.args.visualize:
                    self._visualize_fixed_sample(
                        sample_idx=self.args.visualize_sample_index,
                        epoch=global_epoch,
                        batch_size=self.train_loader.batch_size
                    )

                global_epoch += 1
            # ---------------------------------------------------------
            # ---  Save stage checkpoint after local epochs end ---
            # ---------------------------------------------------------
            stage_filename = f"{current_stage_prefix}_completed"
            self._save_checkpoint(global_epoch, is_best=False, stage_name=stage_filename)
            print(f"\n✅ Stage {stage_idx + 1} complete. Model saved as '{stage_filename}.pth'\n")

        wandb.finish()

        # Create the GIF
        if self.args.visualize:
            from src.visualization import create_sample_gif
            create_sample_gif(sample_idx=self.args.visualize_sample_index)
            for index in range(5):
                create_sample_gif(sample_idx=index)

    # ---------------------------------------------------------------------
    # Visualization Helpers
    # ---------------------------------------------------------------------

    def get_fixed_batch(self, batch_size=100, start_index=0):
        subset_indices = self.train_ds.indices[start_index: start_index + batch_size]
        samples = [self.train_ds.dataset[i] for i in subset_indices]
        batch = graph_scene_collate(samples)
        batch = tuple(x.to(self.device) if torch.is_tensor(x) else x for x in batch)
        return batch

    def _visualize_fixed_sample(self, sample_idx: int, epoch: int, batch_size):
        from src.visualization import visualize_ray_frame

        batch = self.get_fixed_batch(batch_size=batch_size, start_index=0)
        sensor_pos, source_pos, iq_signal, doa_gt = batch

        sensor_pos = sensor_pos.to(self.device)
        source_pos = source_pos.to(self.device)
        iq_signal = iq_signal.to(self.device)
        doa_gt = doa_gt.to(self.device)

        with torch.no_grad():
            model_result = self.model(sensor_pos, iq_signal, doa_gt)
            doa_pred = model_result["bearings"]

        for sample_index in range(5):
            visualize_ray_frame(
                positions=sensor_pos[sample_index],
                bearings=doa_pred[sample_index],
                x_hat=torch.zeros_like(source_pos[sample_index]),
                x_true=source_pos[sample_index],
                step=epoch,
                save_path=f"visualizations/sample_{sample_index:03d}_epoch_{epoch:03d}.png"
            )

            wandb.log({
                "global_epoch": epoch,
                "doa_pred": doa_pred[sample_index],
                "doa_gt": doa_gt[sample_index],
                "pos_gt": source_pos[sample_index]
            })

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
    p.add_argument("--stages_config", type=str, default=None, help="Path to JSON file defining training stages")
    p.add_argument("--val_split", type=float, default=0.1, help="Fraction of data held out for validation")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training hyper‑params (Act as default fallbacks if not in stages config)
    p.add_argument("--train_doa_only", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--val_batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_step_size", type=int, default=20)
    p.add_argument("--lr_gamma", type=float, default=0.2)
    p.add_argument("--val_freq", type=int, default=1, help="Validate every N global epochs")
    p.add_argument(
        "--train_scopes",
        nargs="+",
        default=None,
        help="List of substrings of parameter names to train; others are frozen.",
    )

    # System / misc
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)

    # Visualization
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