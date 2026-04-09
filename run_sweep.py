import os
import sys
import json
import argparse
import csv
import re
import io
import torch
from contextlib import contextmanager, redirect_stdout

import src.create_dataset_script as create_dataset_script
import trainer
import test


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
    """Recursively updates a nested dictionary using a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def parse_and_save_metrics(output_text, csv_path, param_name, param_val, eval_target):
    """Extracts metrics using regex and appends a row to the CSV."""
    total_loss = re.search(r"Combined Total Loss\s+:\s+([\d\.]+)", output_text)
    acc_deg = re.search(r"DOA Accuracy \(Avg/Src\)\s+:\s+([\d\.]+)", output_text)
    net_ue = re.search(r"Network/ESPRIT UE Loss:\s+([\d\.eE+-]+)", output_text)
    true_ue = re.search(r"Theoretical UE Loss\s+:\s+([\d\.eE+-]+)", output_text)

    loss_val = total_loss.group(1) if total_loss else "NaN"
    acc_val = acc_deg.group(1) if acc_deg else "NaN"
    net_ue_val = net_ue.group(1) if net_ue else "NaN"
    true_ue_val = true_ue.group(1) if true_ue else "NaN"

    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([param_name, param_val, eval_target, loss_val, acc_val, net_ue_val, true_ue_val])

    print(f"   -> Saved metrics for [{eval_target}]: Acc={acc_val}°, UE={net_ue_val}")


def main():
    parser = argparse.ArgumentParser(description="Sweep parameters and test ALL stages + ESPRIT.")
    parser.add_argument("--base_config", required=True, help="Path to base JSON config")
    # --- NEW: Accept the stages configuration file ---
    parser.add_argument("--stages_config", type=str, default=None, help="Path to stages JSON config")
    parser.add_argument("--param", required=True, help="Parameter to sweep (e.g., dataset.SNR)")
    parser.add_argument("--values", nargs='+', required=True, help="Values to sweep over")
    parser.add_argument("--out_dir", default="sweeps", help="Directory to store sweep results")
    parser.add_argument("--doa_only", action="store_true", help="Pass this if training DOA only")
    parser.add_argument("--visualize", action="store_true", help="Generate frame visualizations for all stages")
    parser.add_argument("--visualize_sigma", action="store_true", help="Generate Sigma plots for all stages")
    args = parser.parse_args()

    base_config_path = os.path.abspath(args.base_config)
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Load stages config safely into memory if provided
    stages_data = {}
    if args.stages_config:
        stages_config_path = os.path.abspath(args.stages_config)
        with open(stages_config_path, 'r') as f:
            stages_data = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    param_safe_name = args.param.replace('.', '_')
    summary_csv_path = os.path.abspath(os.path.join(args.out_dir, f"{param_safe_name}_sweep_results.csv"))

    with open(summary_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["Sweep_Parameter", "Sweep_Value", "Evaluation_Target", "Total_Loss", "DOA_Accuracy_deg", "Net_UE_Loss",
             "Theoretical_UE_Loss"])

    param_keys = args.param.split('.')
    original_cwd = os.getcwd()

    for val_str in args.values:
        try:
            val = float(val_str) if '.' in val_str else int(val_str)
        except ValueError:
            val = val_str

        print(f"\n{'=' * 70}")
        print(f"🌟 STARTING SWEEP: {args.param} = {val}")
        print(f"{'=' * 70}")

        run_dir_name = f"{param_safe_name}_{val}"
        run_dir = os.path.abspath(os.path.join(args.out_dir, run_dir_name))
        os.makedirs(run_dir, exist_ok=True)

        run_config = base_config.copy()
        update_nested_dict(run_config, param_keys, val)

        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=4)

        # Copy stages_config to the isolated directory if it exists
        local_stages_config = None
        if args.stages_config:
            local_stages_config = os.path.join(run_dir, "stages_config.json")
            with open(local_stages_config, 'w') as f:
                json.dump(stages_data, f, indent=4)

        try:
            os.chdir(run_dir)

            # --- 1. DATA GENERATION ---
            print(f"\n[1/4] Generating Data...")
            data_args = ["create_dataset_script.py", "--config", "config.json", "--output_dir", "."]
            with patch_sys_argv(data_args):
                if hasattr(create_dataset_script, 'main'):
                    create_dataset_script.main()
                else:
                    create_dataset_script.create_dataset()

            # --- 2. TRAIN MODEL (ALL STAGES) ---
            print(f"\n[2/4] Training Model Pipeline...")
            train_args = [
                "trainer.py",
                "--config_path", "config.json",
                "--dataset_path", "dataset.pt",
                "--checkpoint_dir", "."
            ]

            # --- FIXED: explicitly pass stages_config to the trainer ---
            if args.stages_config:
                train_args.extend(["--stages_config", "stages_config.json"])
            else:
                train_args.extend(["--stages_config", "config.json"])  # Fallback if merged

            if args.doa_only: train_args.append("--train_doa_only")

            with patch_sys_argv(train_args):
                trainer.main()

            # --- 3. EVALUATE ALL TRAINED STAGES ---
            print(f"\n[3/4] Evaluating Every Neural Network Stage...")

            # Read stages from the correct source
            if args.stages_config:
                stages = stages_data.get("stages", [{}])
            else:
                stages = run_config.get("stages", [{}])

            for idx, stage in enumerate(stages):
                stage_prefix = stage.get("stage_prefix", f"stage_{idx + 1}")
                checkpoint_to_test = f"best_{stage_prefix}.pth"
                viz_folder_name = f"Stage_{idx + 1}_{stage_prefix}"

                # Extract stage specific criteria params
                loss_func = stage.get("loss_function", "CombinedUncertaintyLoss")
                l_val = stage.get("lambda_val", 0.5)

                print(f"\n  Testing Stage {idx + 1}: {checkpoint_to_test} (Loss: {loss_func}, Lambda: {l_val})")

                test_args = [
                    "test.py",
                    "--config_path", "config.json",
                    "--test_dataset_path", "test_dataset.pt",
                    "--checkpoint_path", checkpoint_to_test,
                    "--viz_prefix", viz_folder_name,
                    "--loss_function", loss_func,
                    "--lambda_val", str(l_val)
                ]
                if args.doa_only: test_args.append("--train_doa_only")
                if args.visualize: test_args.append("--visualize")
                if args.visualize_sigma: test_args.append("--visualize_sigma")

                f_test = io.StringIO()
                with patch_sys_argv(test_args), redirect_stdout(f_test):
                    test.main()

                if torch.cuda.is_available(): torch.cuda.empty_cache()

                test_output = f_test.getvalue()
                parse_and_save_metrics(test_output, summary_csv_path, args.param, val, viz_folder_name)

            # --- 4. EVALUATE ESPRIT BASELINE ---
            print(f"\n[4/4] Evaluating ESPRIT Baseline...")

            esprit_args = [
                "test.py",
                "--config_path", "config.json",
                "--test_dataset_path", "test_dataset.pt",
                "--checkpoint_path", "latest.pth",  # Irrelevant for ESPRIT
                "--esprit_baseline",
                "--viz_prefix", "ESPRIT_Baseline",
                "--loss_function", "CombinedUncertaintyLoss",
                "--lambda_val", "1.0"
            ]
            if args.doa_only: esprit_args.append("--train_doa_only")
            if args.visualize: esprit_args.append("--visualize")
            if args.visualize_sigma: esprit_args.append("--visualize_sigma")

            f_esprit = io.StringIO()
            with patch_sys_argv(esprit_args), redirect_stdout(f_esprit):
                test.main()

            esprit_output = f_esprit.getvalue()
            parse_and_save_metrics(esprit_output, summary_csv_path, args.param, val, "ESPRIT")

        finally:
            os.chdir(original_cwd)

    print(f"\n✅ SWEEP COMPLETE! Summary saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()