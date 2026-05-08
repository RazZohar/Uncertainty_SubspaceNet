import os
import sys
import json
import argparse
import csv
import re
import io
import copy
import torch
from contextlib import contextmanager, redirect_stdout

import src.create_dataset_script as create_dataset_script
import trainer
import test


MODEL_WANDB_PREFIXES = {
    "multi_subarray": "my_model",
    "data_driven_complex": "data_driven",
    "transmusic": "transmusic",
}


@contextmanager
def patch_sys_argv(new_argv):
    """Context manager to safely override sys.argv."""
    original_argv = sys.argv
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = original_argv


def update_nested_dict(d, keys, value):
    """Recursively updates a nested dictionary/list using a list of keys."""
    for key in keys[:-1]:
        if isinstance(d, list):
            key = int(key)
            d = d[key]
        else:
            if key not in d:
                d[key] = {}
            d = d[key]

    last_key = keys[-1]
    if isinstance(d, list):
        d[int(last_key)] = value
    else:
        d[last_key] = value


def parse_and_save_metrics(output_text, csv_path, param_name, param_val, model_type, wandb_prefix, eval_target):
    """Extracts metrics using regex and appends one benchmark row to the CSV."""
    total_loss = re.search(r"Combined Total Loss\s+:\s+([\d\.eE+-]+)", output_text)
    acc_deg = re.search(r"DOA Accuracy \(Avg/Src\)\s+:\s+([\d\.eE+-]+)", output_text)
    net_ue = re.search(r"Network/ESPRIT UE Loss:\s+([\d\.eE+-]+)", output_text)
    true_ue = re.search(r"Theoretical UE Loss\s+:\s+([\d\.eE+-]+)", output_text)
    ccrb_ue = re.search(r"CCRB UE Loss\s+:\s+([\d\.eE+-]+)", output_text)
    ccrb_sigma = re.search(r"CCRB\s+:\s+([\d\.eE+-]+)", output_text)

    row = {
        "Sweep_Parameter": param_name,
        "Sweep_Value": param_val,
        "Model_Type": model_type,
        "WandB_Prefix": wandb_prefix,
        "Evaluation_Target": eval_target,
        "Total_Loss": total_loss.group(1) if total_loss else "NaN",
        "DOA_Accuracy_deg": acc_deg.group(1) if acc_deg else "NaN",
        "Net_UE_Loss": net_ue.group(1) if net_ue else "NaN",
        "Theoretical_UE_Loss": true_ue.group(1) if true_ue else "NaN",
        "CCRB_UE_Loss": ccrb_ue.group(1) if ccrb_ue else "NaN",
        "Mean_CCRB_sigma_deg": ccrb_sigma.group(1) if ccrb_sigma else "NaN",
    }

    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writerow(row)

    print(
        f"   -> Saved metrics for [{model_type}/{eval_target}]: "
        f"Acc={row['DOA_Accuracy_deg']}°, UE={row['Net_UE_Loss']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sweep parameters and benchmark multi-subarray, data-driven, TransMUSIC, and ESPRIT."
    )
    parser.add_argument("--base_config", required=True, help="Path to base JSON config")
    parser.add_argument("--stages_config", type=str, default=None, help="Path to stages JSON config")
    parser.add_argument("--param", required=True, help="Parameter to sweep, e.g. dataset.SNR")
    parser.add_argument("--values", nargs="+", required=True, help="Values to sweep over")
    parser.add_argument("--out_dir", default="sweeps", help="Directory to store sweep results")
    parser.add_argument(
        "--model_types",
        nargs="+",
        choices=list(MODEL_WANDB_PREFIXES.keys()),
        default=["multi_subarray", "data_driven_complex", "transmusic"],
        help="Model types to train/test for every sweep value",
    )
    parser.add_argument("--doa_only", action="store_true", help="Train/evaluate DOA only")
    parser.add_argument("--visualize", action="store_true", help="Generate frame visualizations for all stages")
    parser.add_argument("--visualize_sigma", action="store_true", help="Generate sigma plots for all stages")
    parser.add_argument("--log_eval_to_wandb", action="store_true", help="Also log test runs to W&B")
    parser.add_argument("--wandb_project", type=str, default="multi-subarrays-doa", help="W&B project name")
    args = parser.parse_args()

    base_config_path = os.path.abspath(args.base_config)
    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    stages_data = {}
    if args.stages_config:
        stages_config_path = os.path.abspath(args.stages_config)
        with open(stages_config_path, "r") as f:
            stages_data = json.load(f)

    param_safe_name = args.param.replace(".", "_")
    args.out_dir = os.path.abspath(os.path.join(args.out_dir, f"sweep_{param_safe_name}"))
    os.makedirs(args.out_dir, exist_ok=True)

    summary_csv_path = os.path.abspath(os.path.join(args.out_dir, f"{param_safe_name}_benchmark_results.csv"))
    fieldnames = [
        "Sweep_Parameter",
        "Sweep_Value",
        "Model_Type",
        "WandB_Prefix",
        "Evaluation_Target",
        "Total_Loss",
        "DOA_Accuracy_deg",
        "Net_UE_Loss",
        "Theoretical_UE_Loss",
        "CCRB_UE_Loss",
        "Mean_CCRB_sigma_deg",
    ]
    with open(summary_csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    param_keys = args.param.split(".")
    original_cwd = os.getcwd()

    for val_str in args.values:
        try:
            val = float(val_str) if "." in val_str else int(val_str)
        except ValueError:
            val = val_str

        print(f"\n{'=' * 70}")
        print(f"🌟 STARTING SWEEP: {args.param} = {val}")
        print(f"{'=' * 70}")

        run_dir_name = f"{param_safe_name}_{val}"
        run_dir = os.path.abspath(os.path.join(args.out_dir, run_dir_name))
        os.makedirs(run_dir, exist_ok=True)

        run_config = copy.deepcopy(base_config)
        update_nested_dict(run_config, param_keys, val)

        config_path = os.path.abspath(os.path.join(run_dir, "config.json"))
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=4)

        local_stages_config = None
        if args.stages_config:
            local_stages_config = os.path.abspath(os.path.join(run_dir, "stages_config.json"))
            with open(local_stages_config, "w") as f:
                json.dump(stages_data, f, indent=4)

        try:
            os.chdir(run_dir)

            # 1. Generate one shared dataset per sweep value, then reuse it for all model types.
            print("\n[1/4] Generating shared train/test data...")
            data_args = ["create_dataset_script.py", "--config", config_path, "--output_dir", run_dir]
            with patch_sys_argv(data_args):
                if hasattr(create_dataset_script, "main"):
                    create_dataset_script.main()
                else:
                    create_dataset_script.create_dataset()

            dataset_path = os.path.abspath(os.path.join(run_dir, "dataset.pt"))
            test_dataset_path = os.path.abspath(os.path.join(run_dir, "test_dataset.pt"))

            if args.stages_config:
                stages = stages_data.get("stages", [{}])
            else:
                stages = run_config.get("stages", [{}])
            if not stages:
                stages = [{}]

            # 2-3. Train and evaluate every requested neural benchmark model.
            for model_type in args.model_types:
                model_prefix = MODEL_WANDB_PREFIXES[model_type]
                model_wandb_prefix = f"{model_prefix}_{param_safe_name}_{val}"
                model_dir = os.path.abspath(os.path.join(run_dir, model_type))
                os.makedirs(model_dir, exist_ok=True)

                force_doa_only = args.doa_only or model_type in {"data_driven_complex", "transmusic"}

                print(f"\n[2/4] Training model: {model_type} | W&B prefix: {model_wandb_prefix}")
                train_args = [
                    "trainer.py",
                    "--config_path", config_path,
                    "--dataset_path", dataset_path,
                    "--checkpoint_dir", model_dir,
                    "--model_type", model_type,
                    "--wandb_name_prefix", model_wandb_prefix,
                    "--wandb_project", args.wandb_project,
                ]
                if local_stages_config:
                    train_args.extend(["--stages_config", local_stages_config])
                else:
                    train_args.extend(["--stages_config", config_path])
                if force_doa_only:
                    train_args.append("--train_doa_only")

                with patch_sys_argv(train_args):
                    trainer.main()

                print(f"\n[3/4] Evaluating model: {model_type}")
                for idx, stage in enumerate(stages):
                    stage_prefix = stage.get("stage_prefix", f"stage_{idx + 1}")
                    checkpoint_to_test = os.path.abspath(os.path.join(model_dir, f"best_{stage_prefix}.pth"))
                    if not os.path.exists(checkpoint_to_test):
                        checkpoint_to_test = os.path.abspath(os.path.join(model_dir, "latest.pth"))

                    viz_folder_name = f"{model_type}_Stage_{idx + 1}_{stage_prefix}"
                    loss_func = stage.get("loss_function", "CombinedUncertaintyLoss")
                    l_val = stage.get("lambda_val", 0.5)

                    print(
                        f"\n  Testing {model_type} stage {idx + 1}: "
                        f"{os.path.basename(checkpoint_to_test)} (Loss: {loss_func}, Lambda: {l_val})"
                    )

                    test_args = [
                        "test.py",
                        "--config_path", config_path,
                        "--test_dataset_path", test_dataset_path,
                        "--checkpoint_path", checkpoint_to_test,
                        "--model_type", model_type,
                        "--viz_prefix", viz_folder_name,
                        "--loss_function", loss_func,
                        "--lambda_val", str(l_val),
                        "--wandb_name_prefix", model_wandb_prefix,
                        "--wandb_project", args.wandb_project,
                    ]
                    if force_doa_only:
                        test_args.append("--train_doa_only")
                    if args.log_eval_to_wandb:
                        test_args.append("--log_to_wandb")
                    if args.visualize:
                        test_args.append("--visualize")
                    if args.visualize_sigma:
                        test_args.append("--visualize_sigma")

                    f_test = io.StringIO()
                    with patch_sys_argv(test_args), redirect_stdout(f_test):
                        test.main()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    test_output = f_test.getvalue()
                    print(test_output)
                    parse_and_save_metrics(
                        test_output,
                        summary_csv_path,
                        args.param,
                        val,
                        model_type,
                        model_wandb_prefix,
                        viz_folder_name,
                    )

            # 4. Evaluate ESPRIT once per sweep value on the same test set.
            print("\n[4/4] Evaluating ESPRIT baseline...")
            esprit_args = [
                "test.py",
                "--config_path", config_path,
                "--test_dataset_path", test_dataset_path,
                "--checkpoint_path", os.path.join(run_dir, args.model_types[0], "latest.pth"),
                "--model_type", "multi_subarray",
                "--esprit_baseline",
                "--train_doa_only",
                "--viz_prefix", "ESPRIT_Baseline",
                "--loss_function", "CombinedUncertaintyLoss",
                "--lambda_val", "1.0",
                "--wandb_name_prefix", "esprit_baseline",
                "--wandb_project", args.wandb_project,
            ]
            if args.visualize:
                esprit_args.append("--visualize")
            if args.visualize_sigma:
                esprit_args.append("--visualize_sigma")

            f_esprit = io.StringIO()
            with patch_sys_argv(esprit_args), redirect_stdout(f_esprit):
                test.main()

            esprit_output = f_esprit.getvalue()
            print(esprit_output)
            parse_and_save_metrics(
                esprit_output,
                summary_csv_path,
                args.param,
                val,
                "esprit",
                "esprit_baseline",
                "ESPRIT_Baseline",
            )

        finally:
            os.chdir(original_cwd)

    print(f"\n✅ BENCHMARK SWEEP COMPLETE. Summary saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
