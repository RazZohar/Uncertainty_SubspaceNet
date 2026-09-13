"""SNR sweep evaluator for full-data SSN baselines versus CP-only Stage-1."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from sweeps_scripts.compare_fulltrain_ssn_vs_cp_stage1_vs_stage2 import main as compare_main


def progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_value(text: str) -> Any:
    try:
        return float(text) if "." in text else int(text)
    except ValueError:
        return text


def safe_snr(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def require_file(path: str, label: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return os.path.abspath(path)


def write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate full-data Stage-1/Stage-2 checkpoints against a CP-only Stage-1 sweep"
    )
    parser.add_argument(
        "--full_sweep_root", required=True,
        help="Existing normal full-data sweep root, e.g. sweeps/multi_subarray/sweep_<parameter>",
    )
    parser.add_argument(
        "--cp_sweep_root", required=True,
        help="CP-only sweep root created by run_sweep_cp_stage1_only.py",
    )
    parser.add_argument("--run_prefix", default="subarray_config_0_system_model_snr")
    parser.add_argument("--snr_values", nargs="+", required=True)
    parser.add_argument("--full_checkpoint_subdir", default="multi_subarray")
    parser.add_argument("--cp_checkpoint_subdir", default="cp_stage1")
    parser.add_argument("--full_stage1_checkpoint_name", default="best_base_accuracy.pth")
    parser.add_argument("--stage2_checkpoint_name", default="best_base_reliability.pth")
    parser.add_argument("--cp_stage1_checkpoint_name", default="best_base_accuracy.pth")
    parser.add_argument("--config_name", default="config.json")
    parser.add_argument("--test_dataset_name", default="test_dataset.pt")
    parser.add_argument("--cp_calibration_dataset_name", default="cp_calibration_dataset.pt")
    parser.add_argument("--cp_split_metadata_name", default="cp_stage1_split_metadata.json")
    parser.add_argument(
        "--out_dir",
        default="sweep_results/uq_fulltrain_ssn_vs_cp_stage1_vs_stage2/snr",
    )
    parser.add_argument("--full_stage1_name", default="SubspaceNet Stage 1 (full training data)")
    parser.add_argument("--cp_stage1_name", default="SubspaceNet Stage 1 (80% CP training data)")
    parser.add_argument("--stage2_name", default="SubspaceNet Stage 2 (full training data)")
    parser.add_argument("--include_full_stage1_regular", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_mc", type=int, default=200)
    parser.add_argument("--stddev_spacing", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--progress_every_batches", type=int, default=1)
    parser.add_argument("--mc_progress_every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    args = parse_args(argv)
    full_root = os.path.abspath(args.full_sweep_root)
    cp_root = os.path.abspath(args.cp_sweep_root)
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    completed: List[Any] = []
    progress_path = os.path.join(out_root, "progress.json")
    partial_csv = os.path.join(out_root, "fulltrain_ssn_vs_cp_stage1_vs_stage2_snr_partial.csv")
    partial_json = os.path.join(out_root, "fulltrain_ssn_vs_cp_stage1_vs_stage2_snr_partial.json")
    write_json(progress_path, {
        "status": "running", "current_snr_db": None, "completed_snr_values": completed,
        "total_snr_values": list(args.snr_values), "rows_written": 0,
    })

    for snr_text in args.snr_values:
        snr = parse_value(snr_text)
        full_run = os.path.join(full_root, f"{args.run_prefix}_{snr_text}")
        cp_run = os.path.join(cp_root, f"{args.run_prefix}_{snr_text}")

        config_path = require_file(os.path.join(full_run, args.config_name), "full-data configuration")
        full_stage1_path = require_file(
            os.path.join(full_run, args.full_checkpoint_subdir, args.full_stage1_checkpoint_name),
            "full-data Stage-1 checkpoint",
        )
        stage2_path = require_file(
            os.path.join(full_run, args.full_checkpoint_subdir, args.stage2_checkpoint_name),
            "full-data Stage-2 checkpoint",
        )
        test_path = require_file(os.path.join(full_run, args.test_dataset_name), "independent shared test dataset")
        cp_stage1_path = require_file(
            os.path.join(cp_run, args.cp_checkpoint_subdir, args.cp_stage1_checkpoint_name),
            "CP-only Stage-1 checkpoint",
        )
        cp_cal_path = require_file(
            os.path.join(cp_run, args.cp_calibration_dataset_name),
            "CP-only calibration dataset",
        )
        cp_metadata_path = require_file(
            os.path.join(cp_run, args.cp_split_metadata_name),
            "CP-only split metadata",
        )

        snr_out = os.path.join(out_root, f"SNR_{safe_snr(snr)}")
        os.makedirs(snr_out, exist_ok=True)
        write_json(progress_path, {
            "status": "running", "current_snr_db": snr, "completed_snr_values": completed,
            "total_snr_values": list(args.snr_values), "rows_written": len(all_rows),
        })
        progress("=" * 96)
        progress(f"Full-data vs CP-only Stage-1 evaluation: SNR={snr} dB")
        progress(f"Full-data model run: {full_run}")
        progress(f"CP-only model run: {cp_run}")
        progress(f"Final test dataset: {test_path}")

        compare_argv = [
            "--config_path", config_path,
            "--full_stage1_checkpoint_path", full_stage1_path,
            "--cp_stage1_checkpoint_path", cp_stage1_path,
            "--stage2_checkpoint_path", stage2_path,
            "--cp_calibration_dataset_path", cp_cal_path,
            "--test_dataset_path", test_path,
            "--cp_split_metadata_path", cp_metadata_path,
            "--out_dir", snr_out,
            "--full_stage1_name", args.full_stage1_name,
            "--cp_stage1_name", args.cp_stage1_name,
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
        if args.include_full_stage1_regular:
            compare_argv += ["--include_full_stage1_regular"]

        rows = compare_main(compare_argv)
        for row in rows:
            enriched = dict(row)
            enriched.update({
                "snr_db": snr,
                "full_run_dir": full_run,
                "cp_run_dir": cp_run,
                "full_stage1_checkpoint_path": full_stage1_path,
                "cp_stage1_checkpoint_path": cp_stage1_path,
                "stage2_checkpoint_path": stage2_path,
                "cp_calibration_dataset_path": cp_cal_path,
                "test_dataset_path": test_path,
                "cp_split_metadata_path": cp_metadata_path,
            })
            all_rows.append(enriched)

        write_csv(
            os.path.join(snr_out, "fulltrain_ssn_vs_cp_stage1_vs_stage2_with_snr.csv"),
            [dict(row, snr_db=snr) for row in rows],
        )
        completed.append(snr)
        write_csv(partial_csv, all_rows)
        write_json(partial_json, all_rows)
        write_json(progress_path, {
            "status": "running", "current_snr_db": None, "completed_snr_values": completed,
            "total_snr_values": list(args.snr_values), "rows_written": len(all_rows),
            "last_completed_snr_db": snr,
        })
        progress(f"Completed SNR={snr}; partial rows={len(all_rows)}")

    combined_csv = os.path.join(out_root, "fulltrain_ssn_vs_cp_stage1_vs_stage2_snr_all.csv")
    combined_json = os.path.join(out_root, "fulltrain_ssn_vs_cp_stage1_vs_stage2_snr_all.json")
    combined_pt = os.path.join(out_root, "fulltrain_ssn_vs_cp_stage1_vs_stage2_snr_all.pt")
    write_csv(combined_csv, all_rows)
    write_json(combined_json, all_rows)
    torch.save(all_rows, combined_pt)
    write_json(progress_path, {
        "status": "completed", "current_snr_db": None, "completed_snr_values": completed,
        "total_snr_values": list(args.snr_values), "rows_written": len(all_rows),
    })
    progress(f"Complete. Combined CSV: {combined_csv}")
    return all_rows


if __name__ == "__main__":
    main()
