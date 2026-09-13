#!/usr/bin/env python3
"""
sweep_snr_covariance_comparison.py

SNR-sweep covariance-UQ comparison for the current pipeline.

For every SNR value, the script can:
  1. create a run config with dataset.SNR updated,
  2. generate dataset.pt and test_dataset.pt,
  3. train DataDrivenComplexNet and MultiSubarraysModel, unless checkpoints exist
     and --reuse_existing is set,
  4. compare covariance uncertainty methods:
       - data_driven_regular
       - data_driven_mc_dropout
       - data_driven_conformal_prediction, source-wise CP, Gaussian ±kσ alpha
       - SSN Stage 2
  5. write per-SNR and global CSV/JSON/PT results.

This script is separate from test.py and run_sweep.py. It uses them only as
pipeline references and calls trainer/create_dataset in-process.

Example:
python sweep_snr_covariance_comparison.py \
  --base_config config.json \
  --stages_config stages_config.json \
  --snr_values -3 0 3 10 20 \
  --out_dir snr_cov_uq_sweep \
  --batch_size 1024 \
  --n_mc 200 \
  --stddev_spacing 1.0 \
  --reuse_existing

Notes:
- The multi-subarray checkpoint defaults to best_reliability.pth and is reported as SSN Stage 2.
- Conformal alpha is derived by default from Gaussian two-sided ±stddev_spacing:
    stddev_spacing=1 -> coverage=0.6826894921 -> alpha=0.3173105079
- Source-wise CP uses:
    score[i,s,k] = |theta_true[i,s,k] - mu[i,s,k]| / sqrt(Sigma[i,s,k,k])
  and scales the full covariance by D Sigma D, with D=diag(qhat_1,...,qhat_M).
"""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Copy this script to the project root, next to trainer.py, run_sweep.py, etc.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.create_dataset_script as create_dataset_script
import trainer

from trainer import graph_scene_collate

# This companion script must be in the same directory.
try:
    from sweeps_scripts.compare_datadriven_vs_ssn_stage2_covariance import (
        PredictionSet,
        covariance_metrics,
        deterministic_predictions,
        mc_dropout_predictions,
        conformalize_sourcewise,
        gaussian_alpha_from_stddev_spacing,
        load_data_driven_model,
        load_multi_subarray_model,
        print_table,
        _jsonify,
    )
except Exception as exc:
    raise ImportError(
        "Could not import compare_datadriven_vs_ssn_stage2_covariance.py. "
        "Place both scripts in the project root, or add the script directory to PYTHONPATH."
    ) from exc


MODEL_WANDB_PREFIXES = {
    "data_driven_complex": "data_driven",
    "multi_subarray": "my_model",
}


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

@contextmanager
def patch_sys_argv(new_argv: List[str]):
    old = sys.argv
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = old


def update_nested_dict(d: Any, keys: List[str], value: Any) -> None:
    """Update nested dict/list using keys like ['dataset','SNR']."""
    cur = d
    for key in keys[:-1]:
        if isinstance(cur, list):
            cur = cur[int(key)]
        else:
            cur = cur.setdefault(key, {})
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def safe_name(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("/", "_")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_or_write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_dataset_generation(config_path: str, run_dir: str) -> None:
    """Call the project's dataset generation script in-process."""
    argv = ["create_dataset_script.py", "--config", config_path, "--output_dir", run_dir]
    with patch_sys_argv(argv):
        if hasattr(create_dataset_script, "main"):
            create_dataset_script.main()
        else:
            create_dataset_script.create_dataset()


def call_trainer_direct(train_argv: List[str]) -> None:
    """Call trainer.py in-process and print captured output after the run."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            trainer.main(train_argv)
        except TypeError:
            with patch_sys_argv(["trainer.py", *train_argv]):
                trainer.main()
    print(buf.getvalue())


def checkpoint_exists(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, "latest.pth"))


def resolve_checkpoint_path(
    checkpoint_dir: str,
    preferred_name: str,
    *,
    allow_latest_fallback: bool = False,
) -> str:
    """Return the checkpoint path that should be evaluated."""
    preferred = os.path.join(checkpoint_dir, preferred_name)
    if os.path.isfile(preferred):
        return preferred

    latest = os.path.join(checkpoint_dir, "latest.pth")
    if allow_latest_fallback and os.path.isfile(latest):
        print(
            f"WARNING: preferred checkpoint {preferred} was not found; "
            f"falling back to {latest}."
        )
        return latest

    raise FileNotFoundError(
        f"Missing required checkpoint: {preferred}. "
        f"Set the matching --*_checkpoint_name argument or enable "
        f"--allow_latest_fallback if you intentionally want latest.pth."
    )


# -----------------------------------------------------------------------------
# Train/reuse pipeline
# -----------------------------------------------------------------------------

def train_model_if_needed(
    *,
    args: argparse.Namespace,
    model_type: str,
    config_path: str,
    dataset_path: str,
    checkpoint_dir: str,
    snr_value: Any,
    checkpoint_name: str,
) -> str:
    if args.reuse_existing:
        try:
            ckpt = resolve_checkpoint_path(
                checkpoint_dir,
                checkpoint_name,
                allow_latest_fallback=args.allow_latest_fallback,
            )
            print(f"Reusing existing checkpoint for {model_type}: {ckpt}")
            return ckpt
        except FileNotFoundError:
            pass

    latest_ckpt = os.path.join(checkpoint_dir, "latest.pth")

    ensure_dir(checkpoint_dir)
    wandb_prefix = f"{MODEL_WANDB_PREFIXES[model_type]}_SNR_{safe_name(snr_value)}"

    train_argv = [
        "--config_path", config_path,
        "--dataset_path", dataset_path,
        "--checkpoint_dir", checkpoint_dir,
        "--model_type", model_type,
        "--wandb_project", args.wandb_project,
        "--wandb_name_prefix", wandb_prefix,
        "--batch_size", str(args.train_batch_size),
        "--val_batch_size", str(args.val_batch_size),
        "--num_workers", str(args.num_workers),
        "--tau", str(args.tau),
        "--train_doa_only",
    ]

    if args.stages_config:
        train_argv += ["--stages_config", args.stages_config]
    else:
        train_argv += [
            "--epochs", str(args.epochs),
            "--learning_rate", str(args.learning_rate),
            "--weight_decay", str(args.weight_decay),
            "--loss_function", args.loss_function,
            "--lambda_val", str(args.lambda_val),
        ]

    print(f"\nTraining {model_type} for SNR={snr_value} ...")
    call_trainer_direct(train_argv)

    return resolve_checkpoint_path(
        checkpoint_dir,
        checkpoint_name,
        allow_latest_fallback=True,
    )


# -----------------------------------------------------------------------------
# One-SNR comparison
# -----------------------------------------------------------------------------

def compare_one_snr(
    *,
    args: argparse.Namespace,
    snr_value: Any,
    config_path: str,
    dataset_path: str,
    test_dataset_path: str,
    data_driven_checkpoint_path: str,
    multi_subarray_checkpoint_path: str,
    out_dir: str,
) -> List[Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = args.alpha
    if alpha is None:
        alpha = gaussian_alpha_from_stddev_spacing(args.stddev_spacing)

    print(f"\nComparing covariance methods for SNR={snr_value}")
    print(f"device={device}, n_mc={args.n_mc}, stddev_spacing={args.stddev_spacing}, alpha={alpha}")

    # Add checkpoint paths to the args object expected by loader helpers.
    args_for_load = copy.copy(args)
    args_for_load.config_path = config_path
    args_for_load.data_driven_checkpoint_path = data_driven_checkpoint_path
    args_for_load.multi_subarray_checkpoint_path = multi_subarray_checkpoint_path
    args_for_load.train_doa_only = True

    test_ds = torch.load(test_dataset_path, weights_only=False)
    test_ds.set_use_graph_features(use_features=True)

    cal_ds = torch.load(dataset_path if args.cal_dataset_path is None else args.cal_dataset_path, weights_only=False)
    cal_ds.set_use_graph_features(use_features=True)

    cal_loader = DataLoader(
        cal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_scene_collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_scene_collate,
        drop_last=False,
    )

    data_driven = load_data_driven_model(args_for_load, test_ds, device)
    multi_subarray = load_multi_subarray_model(args_for_load, test_ds, device)

    print("  [1/5] DataDriven regular covariance on calibration set")
    dd_cal = deterministic_predictions(data_driven, cal_loader, device, eps=args.eps)

    print("  [2/5] DataDriven regular covariance on test set")
    dd_test = deterministic_predictions(data_driven, test_loader, device, eps=args.eps)

    print(f"  [3/5] DataDriven MC-Dropout covariance on test set, n_mc={args.n_mc}")
    dd_mc_test = mc_dropout_predictions(data_driven, test_loader, device, n_mc=args.n_mc, eps=args.eps)

    print("  [4/5] DataDriven source-wise conformal covariance")
    dd_cp_test, qhat = conformalize_sourcewise(dd_cal, dd_test, alpha=alpha, eps=args.eps)

    print(f"  [5/5] {args.multi_subarray_method_name} covariance on test set")
    ms_test = deterministic_predictions(multi_subarray, test_loader, device, eps=args.eps)

    rows: List[Dict[str, Any]] = []
    method_sets = {
        "data_driven_regular": dd_test,
        "data_driven_mc_dropout": dd_mc_test,
        "data_driven_conformal_prediction": dd_cp_test,
        args.multi_subarray_method_name: ms_test,
    }

    for method_name, pred_set in method_sets.items():
        row = covariance_metrics(
            method_name,
            pred_set,
            alpha=alpha,
            stddev_spacing=args.stddev_spacing,
            eps=args.eps,
        )
        row["snr"] = snr_value
        row["sweep_parameter"] = args.snr_param
        row["created_at"] = datetime.now().isoformat(timespec="seconds")
        row["data_driven_checkpoint_path"] = data_driven_checkpoint_path
        row["multi_subarray_checkpoint_path"] = multi_subarray_checkpoint_path
        if method_name == "data_driven_conformal_prediction":
            row["qhat"] = _jsonify(qhat.cpu())
            row["base_method"] = "data_driven_regular"
            row["conformal_mode"] = "per_source"
        rows.append(row)

    print_table(rows)

    ensure_dir(out_dir)
    append_or_write_csv(os.path.join(out_dir, "covariance_method_comparison.csv"), rows)
    write_json(os.path.join(out_dir, "covariance_method_comparison.json"), _jsonify(rows))
    torch.save(
        {
            name: {
                "mean_rad": pred.mean_rad,
                "cov_rad2": pred.cov_rad2,
                "target_rad": pred.target_rad,
            }
            for name, pred in method_sets.items()
        },
        os.path.join(out_dir, "covariance_method_tensors.pt"),
    )

    return rows


# -----------------------------------------------------------------------------
# Main SNR sweep
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    parser = argparse.ArgumentParser("SNR sweep for covariance-UQ comparison")
    parser.add_argument("--base_config", required=True, help="Base JSON config used for dataset generation.")
    parser.add_argument("--stages_config", default=None, help="Optional staged training JSON config.")
    parser.add_argument("--snr_values", nargs="+", required=True, help="SNR values to sweep, e.g. -3 0 3 10 20")
    parser.add_argument("--snr_param", default="dataset.SNR", help="Config path for SNR field. Default: dataset.SNR")
    parser.add_argument("--out_dir", default="snr_covariance_uq_sweep")
    parser.add_argument("--results_dir", default=None, help="Optional separate results root. Default: <out_dir>/results_by_snr")

    parser.add_argument("--reuse_existing", action="store_true", help="Reuse existing dataset/checkpoints under out_dir when present.")
    parser.add_argument("--skip_training", action="store_true", help="Do not train. Requires existing checkpoints in each SNR run dir.")
    parser.add_argument("--skip_dataset_generation", action="store_true", help="Do not generate data. Requires existing dataset.pt/test_dataset.pt.")
    parser.add_argument("--data_driven_checkpoint_name", default="latest.pth",
                        help="Checkpoint filename to evaluate under each data_driven_complex directory.")
    parser.add_argument("--multi_subarray_checkpoint_name", default="best_reliability.pth",
                        help="Checkpoint filename to evaluate under each multi_subarray directory. Default loads SSN Stage 2 best_reliability.pth.")
    parser.add_argument("--allow_latest_fallback", action="store_true",
                        help="If the requested checkpoint is missing, allow fallback to latest.pth.")
    parser.add_argument("--multi_subarray_method_name", default="SSN Stage 2",
                        help="Display name used in CSV/JSON for the multi_subarray model row.")

    parser.add_argument("--batch_size", type=int, default=1024, help="Comparison/evaluation batch size.")
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--val_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_mc", type=int, default=200)
    parser.add_argument("--stddev_spacing", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None, help="Override CP alpha. By default derived from Gaussian stddev spacing.")
    parser.add_argument("--cal_dataset_path", default=None, help="Optional fixed calibration dataset. Default: training dataset for each SNR.")
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=42)

    # Trainer fallback args when no stages_config is provided.
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss_function", type=str, default="CombinedUncertaintyLoss")
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--wandb_project", type=str, default="multi-subarrays-doa")

    # Compatibility attributes expected by some constructors.
    parser.add_argument("--train_doa_only", action="store_true", default=True)
    parser.add_argument("--wandb_name_prefix", default=None)

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = ensure_dir(os.path.abspath(args.out_dir))
    results_root = ensure_dir(os.path.abspath(args.results_dir or os.path.join(out_root, "results_by_snr")))
    summary_csv = os.path.join(results_root, "snr_covariance_method_comparison_all.csv")
    summary_json = os.path.join(results_root, "snr_covariance_method_comparison_all.json")
    summary_pt = os.path.join(results_root, "snr_covariance_method_comparison_all.pt")

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    if args.stages_config:
        args.stages_config = os.path.abspath(args.stages_config)

    all_rows: List[Dict[str, Any]] = []
    snr_keys = args.snr_param.split(".")
    original_cwd = os.getcwd()

    for snr_str in args.snr_values:
        try:
            snr_value: Any = float(snr_str) if "." in snr_str else int(snr_str)
        except ValueError:
            snr_value = snr_str

        print("\n" + "=" * 80)
        print(f"SNR sweep value: {snr_value}")
        print("=" * 80)

        run_name = f"SNR_{safe_name(snr_value)}"
        run_dir = ensure_dir(os.path.join(out_root, run_name))
        result_dir = ensure_dir(os.path.join(results_root, run_name))

        run_config = copy.deepcopy(base_config)
        update_nested_dict(run_config, snr_keys, snr_value)
        config_path = os.path.join(run_dir, "config.json")
        write_json(config_path, run_config)
        write_json(os.path.join(result_dir, "config.json"), run_config)

        local_stages_config = None
        if args.stages_config:
            with open(args.stages_config, "r", encoding="utf-8") as f:
                stages_data = json.load(f)
            local_stages_config = os.path.join(run_dir, "stages_config.json")
            write_json(local_stages_config, stages_data)
            write_json(os.path.join(result_dir, "stages_config.json"), stages_data)

        dataset_path = os.path.join(run_dir, "dataset.pt")
        test_dataset_path = os.path.join(run_dir, "test_dataset.pt")

        try:
            os.chdir(run_dir)

            if args.skip_dataset_generation or (args.reuse_existing and os.path.isfile(dataset_path) and os.path.isfile(test_dataset_path)):
                print(f"Reusing dataset files: {dataset_path}, {test_dataset_path}")
            else:
                print("Generating dataset/test_dataset for this SNR...")
                run_dataset_generation(config_path, run_dir)

            if not os.path.isfile(dataset_path) or not os.path.isfile(test_dataset_path):
                raise FileNotFoundError(
                    f"Expected dataset files were not found under {run_dir}: dataset.pt and test_dataset.pt"
                )

            data_driven_dir = ensure_dir(os.path.join(run_dir, "data_driven_complex"))
            multi_subarray_dir = ensure_dir(os.path.join(run_dir, "multi_subarray"))

            args_for_train = copy.copy(args)
            args_for_train.stages_config = local_stages_config

            if args.skip_training:
                dd_ckpt = resolve_checkpoint_path(
                    data_driven_dir,
                    args.data_driven_checkpoint_name,
                    allow_latest_fallback=args.allow_latest_fallback,
                )
                ms_ckpt = resolve_checkpoint_path(
                    multi_subarray_dir,
                    args.multi_subarray_checkpoint_name,
                    allow_latest_fallback=args.allow_latest_fallback,
                )
            else:
                dd_ckpt = train_model_if_needed(
                    args=args_for_train,
                    model_type="data_driven_complex",
                    config_path=config_path,
                    dataset_path=dataset_path,
                    checkpoint_dir=data_driven_dir,
                    snr_value=snr_value,
                    checkpoint_name=args.data_driven_checkpoint_name,
                )
                ms_ckpt = train_model_if_needed(
                    args=args_for_train,
                    model_type="multi_subarray",
                    config_path=config_path,
                    dataset_path=dataset_path,
                    checkpoint_dir=multi_subarray_dir,
                    snr_value=snr_value,
                    checkpoint_name=args.multi_subarray_checkpoint_name,
                )

            rows = compare_one_snr(
                args=args,
                snr_value=snr_value,
                config_path=config_path,
                dataset_path=dataset_path,
                test_dataset_path=test_dataset_path,
                data_driven_checkpoint_path=dd_ckpt,
                multi_subarray_checkpoint_path=ms_ckpt,
                out_dir=result_dir,
            )
            all_rows.extend(rows)
            append_or_write_csv(summary_csv, all_rows)
            write_json(summary_json, _jsonify(all_rows))
            torch.save(_jsonify(all_rows), summary_pt)

        finally:
            os.chdir(original_cwd)

    append_or_write_csv(summary_csv, all_rows)
    write_json(summary_json, _jsonify(all_rows))
    torch.save(_jsonify(all_rows), summary_pt)

    print("\nFinal SNR-sweep outputs:")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")
    print(f"  {summary_pt}")
    return all_rows


if __name__ == "__main__":
    main()
