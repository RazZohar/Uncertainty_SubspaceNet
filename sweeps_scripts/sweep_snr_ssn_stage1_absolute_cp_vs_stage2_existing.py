"""
sweep_snr_ssn_stage1_absolute_cp_vs_stage2_existing.py

Evaluation-only SNR sweep using already-trained SubspaceNet checkpoints.

For every SNR, this driver:
  - loads Stage 1 best_base_accuracy.pth;
  - loads Stage 2 best_base_reliability.pth;
  - splits the already independent test_dataset.pt once, using a reproducible
    random 20% CP-calibration split and an 80% final-evaluation split;
  - compares Stage-1 MC-Dropout / Bayesian diagonal variance, Stage-1 absolute
    residual source-wise CP, and Stage-2 proposed full covariance.

No training and no dataset generation are performed.  dataset.pt is never used
for CP calibration in this workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sweeps_scripts.compare_ssn_stage1_absolute_cp_vs_stage2 import main as compare_main


def progress(message: str) -> None:
    """Emit an unbuffered, timestamped sweep heartbeat."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonify(value), f, indent=2)


def safe_snr(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value


def require_file(path: str, description: str) -> str:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    parser = argparse.ArgumentParser(
        "Evaluate Stage-1 MC/Bayesian/absolute-CP baselines vs Stage-2 covariance over an existing SNR sweep"
    )
    parser.add_argument("--sweep_root", default=None,
                        help="Common root for checkpoints, datasets, and configs.")
    parser.add_argument("--checkpoint_sweep_root", default=None)
    parser.add_argument("--dataset_sweep_root", default=None)
    parser.add_argument("--config_sweep_root", default=None)
    parser.add_argument("--run_prefix", default="subarray_config_0_system_model_snr")
    parser.add_argument("--snr_values", nargs="+", required=True)

    parser.add_argument("--checkpoint_subdir", default="multi_subarray")
    parser.add_argument("--stage1_checkpoint_name", default="best_base_accuracy.pth")
    parser.add_argument("--stage2_checkpoint_name", default="best_base_reliability.pth")
    parser.add_argument("--config_name", default="config.json")
    parser.add_argument("--test_dataset_name", default="test_dataset.pt")
    parser.add_argument(
        "--cp_calibration_fraction",
        type=float,
        default=0.20,
        help=(
            "Random fraction of each independent test_dataset.pt used only to calculate "
            "absolute-residual conformal half-widths. The complement is the final "
            "evaluation set for all methods."
        ),
    )
    parser.add_argument("--out_dir", default="sweep_results/ssn_stage1_baselines_vs_stage2_snr")

    parser.add_argument("--stage1_name", default="SubspaceNet Stage 1")
    parser.add_argument("--stage2_name", default="SubspaceNet Stage 2")
    parser.add_argument("--include_stage1_regular", action="store_true")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_mc", type=int, default=200)
    parser.add_argument("--stddev_spacing", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--progress_every_batches", type=int, default=1)
    parser.add_argument("--mc_progress_every", type=int, default=25)

    args = parser.parse_args(argv)

    common_root = os.path.abspath(args.sweep_root) if args.sweep_root else None
    checkpoint_root = os.path.abspath(args.checkpoint_sweep_root or common_root or ".")
    dataset_root = os.path.abspath(args.dataset_sweep_root or common_root or checkpoint_root)
    config_root = os.path.abspath(args.config_sweep_root or common_root or dataset_root)
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    completed_snrs: List[Any] = []
    progress_json_path = os.path.join(out_root, "progress.json")
    partial_csv_path = os.path.join(out_root, "ssn_stage1_absolute_cp_vs_stage2_snr_partial.csv")
    partial_json_path = os.path.join(out_root, "ssn_stage1_absolute_cp_vs_stage2_snr_partial.json")

    write_json(progress_json_path, {
        "status": "running",
        "current_snr_db": None,
        "completed_snr_values": completed_snrs,
        "total_snr_values": list(args.snr_values),
        "rows_written": 0,
    })
    progress(f"Starting existing-model SNR sweep over {len(args.snr_values)} value(s).")
    progress(f"Live status: {progress_json_path}")
    progress(f"Partial results: {partial_csv_path}")

    for snr_text in args.snr_values:
        try:
            snr_value: Any = float(snr_text) if "." in snr_text else int(snr_text)
        except ValueError:
            snr_value = snr_text

        run_name = f"{args.run_prefix}_{snr_text}"
        checkpoint_run = os.path.join(checkpoint_root, run_name)
        dataset_run = os.path.join(dataset_root, run_name)
        config_run = os.path.join(config_root, run_name)

        stage1_checkpoint_path = require_file(
            os.path.join(checkpoint_run, args.checkpoint_subdir, args.stage1_checkpoint_name),
            "Stage-1/base-accuracy checkpoint",
        )
        stage2_checkpoint_path = require_file(
            os.path.join(checkpoint_run, args.checkpoint_subdir, args.stage2_checkpoint_name),
            "Stage-2/base-reliability checkpoint",
        )
        test_dataset_path = require_file(
            os.path.join(dataset_run, args.test_dataset_name),
            "test dataset",
        )
        config_path = require_file(
            os.path.join(config_run, args.config_name),
            "run configuration",
        )

        snr_out = os.path.join(out_root, f"SNR_{safe_snr(snr_value)}")
        os.makedirs(snr_out, exist_ok=True)

        write_json(progress_json_path, {
            "status": "running",
            "current_snr_db": snr_value,
            "completed_snr_values": completed_snrs,
            "total_snr_values": list(args.snr_values),
            "rows_written": len(all_rows),
        })
        progress("=" * 90)
        progress(f"Starting SNR = {snr_value} dB")
        progress(f"Stage 1 checkpoint: {stage1_checkpoint_path}")
        progress(f"Stage 2 checkpoint: {stage2_checkpoint_path}")
        progress(f"Independent test dataset: {test_dataset_path}")
        progress(
            f"CP calibration: random {100.0 * args.cp_calibration_fraction:.1f}% split of this test dataset; "
            f"all methods are evaluated on the remaining "
            f"{100.0 * (1.0 - args.cp_calibration_fraction):.1f}%."
        )
        progress("=" * 90)

        compare_argv = [
            "--config_path", config_path,
            "--stage1_checkpoint_path", stage1_checkpoint_path,
            "--stage2_checkpoint_path", stage2_checkpoint_path,
            "--test_dataset_path", test_dataset_path,
            "--cp_calibration_fraction", str(args.cp_calibration_fraction),
            "--out_dir", snr_out,
            "--stage1_name", args.stage1_name,
            "--stage2_name", args.stage2_name,
            "--batch_size", str(args.batch_size),
            "--num_workers", str(args.num_workers),
            "--n_mc", str(args.n_mc),
            "--stddev_spacing", str(args.stddev_spacing),
            "--seed", str(args.seed),
            "--eps", str(args.eps),
            "--progress_every_batches", str(args.progress_every_batches),
            "--mc_progress_every", str(args.mc_progress_every),
        ]
        if args.alpha is not None:
            compare_argv += ["--alpha", str(args.alpha)]
        if args.include_stage1_regular:
            compare_argv += ["--include_stage1_regular"]

        rows = compare_main(compare_argv)
        for row in rows:
            enriched = dict(row)
            enriched["snr_db"] = snr_value
            enriched["stage1_checkpoint_path"] = stage1_checkpoint_path
            enriched["stage2_checkpoint_path"] = stage2_checkpoint_path
            enriched["config_path"] = config_path
            enriched["test_dataset_path"] = test_dataset_path
            enriched["cp_calibration_fraction"] = float(args.cp_calibration_fraction)
            enriched["cp_calibration_source"] = "random split of independent test_dataset.pt"
            all_rows.append(enriched)

        write_csv(
            os.path.join(snr_out, "ssn_stage1_absolute_cp_vs_stage2_with_snr.csv"),
            [dict(row, snr_db=snr_value) for row in rows],
        )
        completed_snrs.append(snr_value)
        write_csv(partial_csv_path, all_rows)
        write_json(partial_json_path, all_rows)
        write_json(progress_json_path, {
            "status": "running",
            "current_snr_db": None,
            "completed_snr_values": completed_snrs,
            "total_snr_values": list(args.snr_values),
            "rows_written": len(all_rows),
            "last_completed_snr_db": snr_value,
        })
        progress(
            f"Finished SNR = {snr_value} dB. "
            f"Partial merged results now contain {len(all_rows)} method rows: {partial_csv_path}"
        )

    combined_csv = os.path.join(out_root, "ssn_stage1_absolute_cp_vs_stage2_snr_all.csv")
    combined_json = os.path.join(out_root, "ssn_stage1_absolute_cp_vs_stage2_snr_all.json")
    combined_pt = os.path.join(out_root, "ssn_stage1_absolute_cp_vs_stage2_snr_all.pt")

    write_csv(combined_csv, all_rows)
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(jsonify(all_rows), f, indent=2)
    torch.save(all_rows, combined_pt)

    write_json(progress_json_path, {
        "status": "completed",
        "current_snr_db": None,
        "completed_snr_values": completed_snrs,
        "total_snr_values": list(args.snr_values),
        "rows_written": len(all_rows),
    })
    progress("All SNR values completed.")
    progress(f"Combined CSV: {combined_csv}")
    progress(f"Combined JSON: {combined_json}")
    progress(f"Combined tensors: {combined_pt}")
    return all_rows


if __name__ == "__main__":
    main()
