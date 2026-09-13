"""
Train only a dedicated Stage-1 SubspaceNet model for the split-conformal baseline.

This script intentionally DOES NOT train, modify, or replace the standard models.

Expected existing full-data sweep tree (created by the normal sweep script):
  sweeps/multi_subarray/sweep_<parameter>/<parameter>_<snr>/
      dataset.pt              # used by standard models in full
      test_dataset.pt         # independent final test set
      config.json
      multi_subarray/
          best_base_accuracy.pth
          best_base_reliability.pth

For every SNR, this script:
  1. reads the existing dataset.pt;
  2. creates a dedicated CP-only 80%/20% split of that dataset;
  3. trains ONLY a new Stage-1 multi_subarray model on the CP 80% split;
  4. keeps the 20% CP split untouched for estimating q_alpha,k;
  5. leaves all full-data models and their checkpoints unchanged.

The full-data test_dataset.pt is only referenced, not regenerated or consumed in
training. It remains the common independent final test set for every method.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch

import trainer_strict_cp as trainer
from strict_cp_dataset_subset import FileBackedSceneSubset


def progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@contextmanager
def patch_sys_argv(new_argv: List[str]):
    original = sys.argv
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = original


def call_trainer(train_argv: List[str]) -> None:
    """Support both trainer.main(argv) and legacy trainer.main() entry points."""
    try:
        trainer.main(train_argv)
    except TypeError:
        with patch_sys_argv(["trainer_strict_cp.py", *train_argv]):
            trainer.main()


def safe_snr(value: Any) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p")


def parse_value(text: str) -> Any:
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def load_stages_stage1_only(stages_config_path: str) -> Dict[str, Any]:
    """Extract the base-accuracy stage; fall back to the first configured stage."""
    with open(stages_config_path, "r", encoding="utf-8") as f:
        stages_data = json.load(f)

    stages = stages_data.get("stages", [])
    if not isinstance(stages, list) or not stages:
        raise ValueError(
            f"{stages_config_path} must contain a non-empty top-level 'stages' list."
        )

    selected: Optional[Dict[str, Any]] = None
    for stage in stages:
        if str(stage.get("stage_prefix", "")).strip() == "base_accuracy":
            selected = copy.deepcopy(stage)
            break
    if selected is None:
        selected = copy.deepcopy(stages[0])

    # The CP model is explicitly a Stage-1/accuracy model. Preserve the existing
    # stage settings (epochs, LR, batch size, loss), but force the expected name.
    selected["stage_prefix"] = "base_accuracy"
    return {"stages": [selected]}


def make_or_load_cp_split(
    dataset_path: str,
    cp_run_dir: str,
    train_fraction: float,
    split_seed: int,
    overwrite: bool,
    full_run_dir: str,
    test_dataset_path: str,
) -> Dict[str, Any]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("cp_train_fraction must be strictly between 0 and 1.")

    train_subset_path = os.path.join(cp_run_dir, "cp_train_dataset.pt")
    cal_subset_path = os.path.join(cp_run_dir, "cp_calibration_dataset.pt")
    metadata_path = os.path.join(cp_run_dir, "cp_stage1_split_metadata.json")

    if (
        not overwrite
        and os.path.isfile(train_subset_path)
        and os.path.isfile(cal_subset_path)
        and os.path.isfile(metadata_path)
    ):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        progress("Reusing existing CP-only train/calibration split.")
        return metadata

    full_dataset = torch.load(dataset_path, weights_only=False)
    n_total = len(full_dataset)
    if n_total < 2:
        raise ValueError("dataset.pt needs at least two scenes to create CP train/calibration splits.")

    n_train = int(round(n_total * train_fraction))
    n_train = min(max(1, n_train), n_total - 1)
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(n_total, generator=generator).tolist()
    train_indices = [int(i) for i in permutation[:n_train]]
    calibration_indices = [int(i) for i in permutation[n_train:]]

    cp_train = FileBackedSceneSubset(
        dataset_path=dataset_path,
        indices=train_indices,
        split_name="cp_stage1_train",
    )
    cp_calibration = FileBackedSceneSubset(
        dataset_path=dataset_path,
        indices=calibration_indices,
        split_name="cp_calibration",
    )
    torch.save(cp_train, train_subset_path)
    torch.save(cp_calibration, cal_subset_path)
    torch.save(
        {
            "cp_train_indices": train_indices,
            "cp_calibration_indices": calibration_indices,
        },
        os.path.join(cp_run_dir, "cp_stage1_split_indices.pt"),
    )

    metadata: Dict[str, Any] = {
        "protocol": "dedicated_cp_stage1_only",
        "source_full_data_run_dir": os.path.abspath(full_run_dir),
        "source_dataset_path": os.path.abspath(dataset_path),
        "source_test_dataset_path": os.path.abspath(test_dataset_path),
        "cp_run_dir": os.path.abspath(cp_run_dir),
        "cp_train_dataset_path": os.path.abspath(train_subset_path),
        "cp_calibration_dataset_path": os.path.abspath(cal_subset_path),
        "cp_train_fraction_requested": float(train_fraction),
        "cp_train_num_scenes": int(n_train),
        "cp_calibration_num_scenes": int(n_total - n_train),
        "total_training_dataset_num_scenes": int(n_total),
        "split_seed": int(split_seed),
        "cp_train_indices": train_indices,
        "cp_calibration_indices": calibration_indices,
        "guarantees": {
            "cp_stage1_model_trains_only_on_cp_train_dataset": True,
            "cp_calibration_unused_by_cp_model_fitting_validation_or_checkpoint_selection": True,
            "standard_full_data_models_unchanged": True,
            "final_test_dataset_unused_by_all_training": True,
        },
    }
    write_json(metadata_path, metadata)
    return metadata


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train a CP-only Stage-1 model while preserving all existing full-data models"
    )
    parser.add_argument(
        "--full_sweep_root",
        required=True,
        help=(
            "Existing normal sweep root, e.g. "
            "sweeps/multi_subarray/sweep_subarray_config_0_system_model_snr"
        ),
    )
    parser.add_argument(
        "--cp_sweep_root",
        default="sweeps/cp_stage1_only",
        help="Root in which CP-only train/calibration splits and checkpoint are stored.",
    )
    parser.add_argument("--run_prefix", default="subarray_config_0_system_model_snr")
    parser.add_argument("--snr_values", nargs="+", required=True)
    parser.add_argument(
        "--stages_config",
        required=True,
        help="Normal multi-stage training JSON; this script extracts only its Stage-1/base-accuracy stage.",
    )
    parser.add_argument("--cp_train_fraction", type=float, default=0.80)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--trainer_val_split",
        type=float,
        default=0.10,
        help=(
            "Internal validation fraction drawn only from the CP 80% training partition. "
            "The CP 20% calibration partition remains untouched."
        ),
    )
    parser.add_argument("--trainer_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_project", default="multi-subarrays-doa")
    parser.add_argument("--dataset_name", default="dataset.pt")
    parser.add_argument("--test_dataset_name", default="test_dataset.pt")
    parser.add_argument("--config_name", default="config.json")
    parser.add_argument("--full_checkpoint_subdir", default="multi_subarray")
    parser.add_argument(
        "--overwrite_split",
        action="store_true",
        help="Regenerate CP train/calibration indices even when an existing split is present.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    args = parse_args(argv)
    if not (0.0 < args.cp_train_fraction < 1.0):
        raise ValueError("cp_train_fraction must be strictly between 0 and 1.")
    if not (0.0 < args.trainer_val_split < 1.0):
        raise ValueError("trainer_val_split must be strictly between 0 and 1.")

    full_sweep_root = os.path.abspath(args.full_sweep_root)
    cp_sweep_root = os.path.abspath(
        os.path.join(args.cp_sweep_root, f"sweep_{args.run_prefix}")
        if os.path.basename(os.path.normpath(args.cp_sweep_root)) != f"sweep_{args.run_prefix}"
        else args.cp_sweep_root
    )
    os.makedirs(cp_sweep_root, exist_ok=True)
    stages_config_path = os.path.abspath(args.stages_config)
    stage1_only_config = load_stages_stage1_only(stages_config_path)

    manifest: List[Dict[str, Any]] = []
    for snr_text in args.snr_values:
        snr_value = parse_value(snr_text)
        full_run_dir = os.path.join(full_sweep_root, f"{args.run_prefix}_{snr_text}")
        cp_run_dir = os.path.join(cp_sweep_root, f"{args.run_prefix}_{snr_text}")
        os.makedirs(cp_run_dir, exist_ok=True)

        dataset_path = os.path.join(full_run_dir, args.dataset_name)
        test_dataset_path = os.path.join(full_run_dir, args.test_dataset_name)
        source_config_path = os.path.join(full_run_dir, args.config_name)
        full_stage1_path = os.path.join(full_run_dir, args.full_checkpoint_subdir, "best_base_accuracy.pth")
        full_stage2_path = os.path.join(full_run_dir, args.full_checkpoint_subdir, "best_base_reliability.pth")

        required = {
            "full training dataset": dataset_path,
            "independent final test dataset": test_dataset_path,
            "configuration": source_config_path,
            "full-data Stage-1 checkpoint": full_stage1_path,
            "full-data Stage-2 checkpoint": full_stage2_path,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                "The normal full-data sweep must complete before the CP-only training job.\n"
                + "\n".join(missing)
            )

        progress("=" * 96)
        progress(f"CP-only Stage-1 training: SNR={snr_value} dB")
        progress(f"Full-data models remain at: {full_run_dir}")
        progress(f"CP-only artifacts write to: {cp_run_dir}")

        # Copy config for a self-contained CP run. The data themselves remain file-backed.
        cp_config_path = os.path.join(cp_run_dir, "config.json")
        shutil.copy2(source_config_path, cp_config_path)
        cp_stage1_stages_path = os.path.join(cp_run_dir, "cp_stage1_stages_config.json")
        write_json(cp_stage1_stages_path, stage1_only_config)

        metadata = make_or_load_cp_split(
            dataset_path=dataset_path,
            cp_run_dir=cp_run_dir,
            train_fraction=args.cp_train_fraction,
            split_seed=args.split_seed,
            overwrite=args.overwrite_split,
            full_run_dir=full_run_dir,
            test_dataset_path=test_dataset_path,
        )
        progress(
            f"CP-only split: train={metadata['cp_train_num_scenes']} "
            f"({100.0 * args.cp_train_fraction:.1f}%), calibration={metadata['cp_calibration_num_scenes']} "
            f"({100.0 * (1.0 - args.cp_train_fraction):.1f}%)."
        )
        progress(
            "The standard Stage-1, Stage-2, and all other normal benchmark models remain trained "
            "with the original full dataset.pt."
        )

        cp_checkpoint_dir = os.path.join(cp_run_dir, "cp_stage1")
        os.makedirs(cp_checkpoint_dir, exist_ok=True)
        train_argv = [
            "--dataset_path", metadata["cp_train_dataset_path"],
            "--config_path", cp_config_path,
            "--stages_config", cp_stage1_stages_path,
            "--checkpoint_dir", cp_checkpoint_dir,
            "--model_type", "multi_subarray",
            "--wandb_name_prefix", f"cp_stage1_{args.run_prefix}_{snr_text}",
            "--wandb_project", args.wandb_project,
            "--val_split", str(args.trainer_val_split),
            "--seed", str(args.trainer_seed),
            "--num_workers", str(args.num_workers),
            "--train_doa_only",
        ]
        progress(
            "Training only CP-specific Stage-1/base_accuracy checkpoint. "
            f"Its validation split is {100.0 * args.trainer_val_split:.1f}% of its 80% CP training partition."
        )
        call_trainer(train_argv)

        cp_checkpoint_path = os.path.join(cp_checkpoint_dir, "best_base_accuracy.pth")
        if not os.path.isfile(cp_checkpoint_path):
            raise FileNotFoundError(
                "CP-only trainer did not create expected checkpoint: " + cp_checkpoint_path
            )

        record = {
            "snr_db": snr_value,
            "full_run_dir": os.path.abspath(full_run_dir),
            "cp_run_dir": os.path.abspath(cp_run_dir),
            "full_stage1_checkpoint_path": os.path.abspath(full_stage1_path),
            "full_stage2_checkpoint_path": os.path.abspath(full_stage2_path),
            "cp_stage1_checkpoint_path": os.path.abspath(cp_checkpoint_path),
            "cp_calibration_dataset_path": metadata["cp_calibration_dataset_path"],
            "test_dataset_path": os.path.abspath(test_dataset_path),
            "cp_stage1_train_fraction": args.cp_train_fraction,
            "cp_calibration_fraction": 1.0 - args.cp_train_fraction,
            "trainer_val_split_within_cp_train": args.trainer_val_split,
            "split_seed": args.split_seed,
            "protocol": (
                "Only the CP-specific Stage-1 model was trained on the 80% CP training split. "
                "All regular benchmark models retain their normal full dataset.pt training."
            ),
        }
        manifest.append(record)
        write_json(os.path.join(cp_run_dir, "cp_stage1_run_manifest.json"), record)
        progress(f"CP-only Stage-1 checkpoint complete: {cp_checkpoint_path}")

    manifest_path = os.path.join(cp_sweep_root, "cp_stage1_only_training_manifest.json")
    write_json(manifest_path, {"runs": manifest})
    progress(f"Complete. Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    main()
