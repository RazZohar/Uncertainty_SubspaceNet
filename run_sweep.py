import os
import sys
import json
import argparse
import csv
import re
import io
import copy
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import src.create_dataset_script as create_dataset_script
import trainer
import test


MODEL_WANDB_PREFIXES = {
    "multi_subarray": "my_model",
    "data_driven_complex": "data_driven",
    "transmusic": "transmusic",
}

UQ_METRIC_COLUMNS = [
    "ANEES_Raw",
    "ANEES_Normalized",
    "Log_ANEES_Normalized",
    "APEC_Trace",
    "EEC_Trace",
    "APEC_EEC_Trace_Gap",
    "APEC_EEC_Fro",
    "APEC_EEC_Rel",
]


@contextmanager
def patch_sys_argv(new_argv):
    """Fallback context manager for legacy modules that still parse sys.argv."""
    original_argv = sys.argv
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = original_argv


def update_nested_dict(d, keys, value):
    """Recursively update a nested dictionary/list using a list of keys."""
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


def _regex_value(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else "NaN"


def _safe_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, int, str, bool)):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value
    try:
        return float(value)
    except Exception:
        return value


def parse_metrics_from_text(output_text: str) -> Dict[str, Any]:
    """Extract summary metrics from the printed test.py report."""
    return {
        "Total_Loss": _safe_float(_regex_value(r"Combined Total Loss\s+:\s+([\d\.eE+-]+)", output_text)),
        "DOA_Accuracy_deg": _safe_float(_regex_value(r"DOA Accuracy \(Avg/Src\)\s+:\s+([\d\.eE+-]+)", output_text)),
        "Net_UE_Loss": _safe_float(_regex_value(r"Network/ESPRIT UE Loss:\s+([\d\.eE+-]+)", output_text)),
        "Theoretical_UE_Loss": _safe_float(_regex_value(r"Theoretical UE Loss\s+:\s+([\d\.eE+-]+)", output_text)),
        "CCRB_UE_Loss": _safe_float(_regex_value(r"CCRB UE Loss\s+:\s+([\d\.eE+-]+)", output_text)),
        "Mean_CCRB_sigma_deg": _safe_float(_regex_value(r"CCRB\s+:\s+([\d\.eE+-]+)", output_text)),
        "ANEES_Raw": _safe_float(_regex_value(r"ANEES raw\s+:\s+([\d\.eE+-]+)", output_text)),
        "ANEES_Normalized": _safe_float(_regex_value(r"ANEES normalized\s+:\s+([\d\.eE+-]+)", output_text)),
        "Log_ANEES_Normalized": _safe_float(_regex_value(r"log\(ANEES norm\.\)\s+:\s+([\d\.eE+-]+)", output_text)),
        "APEC_Trace": _safe_float(_regex_value(r"APEC trace\s+:\s+([\d\.eE+-]+)", output_text)),
        "EEC_Trace": _safe_float(_regex_value(r"EEC trace\s+:\s+([\d\.eE+-]+)", output_text)),
        "APEC_EEC_Trace_Gap": _safe_float(_regex_value(r"APEC-EEC trace gap\s+:\s+([\d\.eE+-]+)", output_text)),
        "APEC_EEC_Fro": _safe_float(_regex_value(r"\|\|APEC-EEC\|\|_F\s+:\s+([\d\.eE+-]+)", output_text)),
        "APEC_EEC_Rel": _safe_float(_regex_value(r"rel \|\|APEC-EEC\|\|_F\s+:\s+([\d\.eE+-]+)", output_text)),
    }


def _to_plain_python(value: Any) -> Any:
    """Convert tensors / numpy scalars to plain Python objects for pickle-safe npy records."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    return value


def build_metrics_from_returned(test_metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a readable summary from the dictionary returned by test.main()."""
    if not test_metrics:
        return {}

    m = _to_plain_python(test_metrics)
    summary = {
        "Total_Loss": m.get("total_loss"),
        "Net_UE_Loss": m.get("net_ue_loss"),
        "Theoretical_UE_Loss": m.get("true_ue_loss"),
        "CCRB_UE_Loss": m.get("ccrb_ue_loss"),
        "Mean_CCRB_sigma_deg": m.get("ccrb_value"),
    }

    try:
        summary["DOA_Accuracy_deg"] = float(np.degrees(np.sqrt(m["rmspe_sq"] / m["num_sources"])))
    except Exception:
        summary["DOA_Accuracy_deg"] = None

    # Prefer the returned test.py dictionary over regex parsing for the UQ metrics.
    # The printed report is rounded, while these values preserve full precision.
    num_sources = m.get("num_sources", 1) or 1
    anees_norm = m.get("anees_normalized", m.get("anees"))
    anees_raw = m.get("anees_raw")
    if anees_raw is None and anees_norm is not None:
        anees_raw = anees_norm * num_sources

    uq_map = {
        "ANEES_Raw": anees_raw,
        "ANEES_Normalized": anees_norm,
        "Log_ANEES_Normalized": m.get("log_anees_normalized", m.get("log_anees")),
        "APEC_Trace": m.get("apec_trace"),
        "EEC_Trace": m.get("eec_trace"),
        "APEC_EEC_Trace_Gap": m.get("apec_eec_trace_gap"),
        "APEC_EEC_Fro": m.get("apec_eec_fro"),
        "APEC_EEC_Rel": m.get("apec_eec_rel"),
    }
    for key, value in uq_map.items():
        summary[key] = _safe_float(value)

    return summary


def merge_metrics(test_metrics: Optional[Dict[str, Any]], output_text: str) -> Dict[str, Any]:
    """Use full-precision values returned by test.main(), with stdout parsing as fallback."""
    parsed = parse_metrics_from_text(output_text)
    returned = build_metrics_from_returned(test_metrics)
    merged = dict(parsed)
    for key, value in returned.items():
        if value is not None:
            merged[key] = value
    return merged


def write_csv_row(
    csv_path: str,
    param_name: str,
    param_val: Any,
    model_type: str,
    wandb_prefix: str,
    eval_target: str,
    parsed_metrics: Dict[str, Any],
):
    """Append one benchmark row to the CSV, preserving the previous text-parsed behavior."""
    row = {
        "Sweep_Parameter": param_name,
        "Sweep_Value": param_val,
        "Model_Type": model_type,
        "WandB_Prefix": wandb_prefix,
        "Evaluation_Target": eval_target,
        "Total_Loss": parsed_metrics.get("Total_Loss", "NaN"),
        "DOA_Accuracy_deg": parsed_metrics.get("DOA_Accuracy_deg", "NaN"),
        "Net_UE_Loss": parsed_metrics.get("Net_UE_Loss", "NaN"),
        "Theoretical_UE_Loss": parsed_metrics.get("Theoretical_UE_Loss", "NaN"),
        "CCRB_UE_Loss": parsed_metrics.get("CCRB_UE_Loss", "NaN"),
        "Mean_CCRB_sigma_deg": parsed_metrics.get("Mean_CCRB_sigma_deg", "NaN"),
        "ANEES_Raw": parsed_metrics.get("ANEES_Raw", "NaN"),
        "ANEES_Normalized": parsed_metrics.get("ANEES_Normalized", "NaN"),
        "Log_ANEES_Normalized": parsed_metrics.get("Log_ANEES_Normalized", "NaN"),
        "APEC_Trace": parsed_metrics.get("APEC_Trace", "NaN"),
        "EEC_Trace": parsed_metrics.get("EEC_Trace", "NaN"),
        "APEC_EEC_Trace_Gap": parsed_metrics.get("APEC_EEC_Trace_Gap", "NaN"),
        "APEC_EEC_Fro": parsed_metrics.get("APEC_EEC_Fro", "NaN"),
        "APEC_EEC_Rel": parsed_metrics.get("APEC_EEC_Rel", "NaN"),
    }

    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writerow(row)

    print(
        f"   -> Saved CSV metrics for [{model_type}/{eval_target}]: "
        f"Acc={row['DOA_Accuracy_deg']}°, UE={row['Net_UE_Loss']}, "
        f"ANEES={row['ANEES_Normalized']}, APEC/EEC rel={row['APEC_EEC_Rel']}"
    )
    return row


def save_results_npy(records: List[Dict[str, Any]], *paths: str):
    """Save all accumulated full-result records as a numpy object array."""
    arr = np.array(records, dtype=object)
    for path in paths:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, arr, allow_pickle=True)


def run_dataset_generation(config_path: str, run_dir: str):
    """Call dataset generation in-process. Falls back to sys.argv because that script is legacy CLI-only."""
    data_args = ["create_dataset_script.py", "--config", config_path, "--output_dir", run_dir]
    with patch_sys_argv(data_args):
        if hasattr(create_dataset_script, "main"):
            create_dataset_script.main()
        else:
            create_dataset_script.create_dataset()


def call_trainer_direct(train_argv: List[str]):
    """Call trainer.py directly as Python code, not through subprocess."""
    try:
        return trainer.main(train_argv)
    except TypeError:
        # Backward-compatible fallback if an older trainer.py has main() without argv.
        with patch_sys_argv(["trainer.py", *train_argv]):
            return trainer.main()


def call_test_direct(test_argv: List[str]) -> tuple[Optional[Dict[str, Any]], str]:
    """Call test.py directly and capture both its returned metrics and printed report."""
    f_test = io.StringIO()
    test_metrics = None
    with redirect_stdout(f_test):
        try:
            test_metrics = test.main(argv=test_argv)
        except TypeError:
            # Backward-compatible fallback if an older test.py has main() without argv.
            with patch_sys_argv(["test.py", *test_argv]):
                test_metrics = test.main()
    return test_metrics, f_test.getvalue()


def append_full_record(
    records: List[Dict[str, Any]],
    *,
    param_name: str,
    param_val: Any,
    model_type: str,
    wandb_prefix: str,
    eval_target: str,
    checkpoint_path: str,
    config_path: str,
    dataset_path: str,
    test_dataset_path: str,
    loss_function: str,
    lambda_val: float,
    test_metrics: Optional[Dict[str, Any]],
    output_text: str,
    parsed_row: Dict[str, Any],
    output_text_path: str,
):
    returned_summary = build_metrics_from_returned(test_metrics)
    parsed_metrics = merge_metrics(test_metrics, output_text)

    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sweep_parameter": param_name,
        "sweep_value": param_val,
        "model_type": model_type,
        "wandb_prefix": wandb_prefix,
        "evaluation_target": eval_target,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "dataset_path": dataset_path,
        "test_dataset_path": test_dataset_path,
        "loss_function": loss_function,
        "lambda_val": lambda_val,
        "metrics_returned_by_test_main": _to_plain_python(test_metrics) if test_metrics is not None else None,
        "metrics_summary_from_return": returned_summary,
        "metrics_parsed_from_stdout": parsed_metrics,
        "csv_row": parsed_row,
        "stdout_text_path": output_text_path,
        "stdout_text": output_text,
    }
    records.append(record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep parameters and benchmark multi-subarray, data-driven, TransMUSIC, and ESPRIT."
    )
    parser.add_argument("--base_config", required=True, help="Path to base JSON config")
    parser.add_argument("--stages_config", type=str, default=None, help="Path to stages JSON config")
    parser.add_argument("--param", required=True, help="Parameter to sweep, e.g. dataset.SNR")
    parser.add_argument("--values", nargs="+", required=True, help="Values to sweep over")
    parser.add_argument("--out_dir", default="sweeps", help="Directory to store working sweep runs, datasets, and checkpoints")
    parser.add_argument(
        "--results_dir",
        default=None,
        help=(
            "Dedicated directory for exported sweep results (.npy, per-value CSV, "
            "and captured evaluation text). Defaults to <out_dir>/results_by_value."
        ),
    )
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
    args = parser.parse_args(argv)

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

    # Global summaries stay directly under the sweep directory.
    summary_csv_path = os.path.abspath(os.path.join(args.out_dir, f"{param_safe_name}_benchmark_results.csv"))
    summary_npy_path = os.path.abspath(os.path.join(args.out_dir, f"{param_safe_name}_benchmark_results.npy"))

    # Per-sweep-value exported results are kept outside the run_dir so the working
    # dataset/checkpoint tree and the analysis/result tree are separated.
    results_root_dir = os.path.abspath(args.results_dir or os.path.join(args.out_dir, "results_by_value"))
    os.makedirs(results_root_dir, exist_ok=True)

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
        *UQ_METRIC_COLUMNS,
    ]
    with open(summary_csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    all_records: List[Dict[str, Any]] = []
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

        # Dedicated exported-results directory for this sweep value.
        # This is intentionally NOT inside run_dir.
        value_results_dir = os.path.abspath(os.path.join(results_root_dir, run_dir_name))
        os.makedirs(value_results_dir, exist_ok=True)
        run_results_npy_path = os.path.abspath(os.path.join(value_results_dir, "full_sweep_results.npy"))
        value_csv_path = os.path.abspath(os.path.join(value_results_dir, "benchmark_results.csv"))
        text_outputs_dir = os.path.abspath(os.path.join(value_results_dir, "evaluation_text_outputs"))
        os.makedirs(text_outputs_dir, exist_ok=True)

        with open(value_csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

        value_records: List[Dict[str, Any]] = []

        run_config = copy.deepcopy(base_config)
        update_nested_dict(run_config, param_keys, val)

        config_path = os.path.abspath(os.path.join(run_dir, "config.json"))
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=4)

        # Mirror only lightweight run metadata/config into the dedicated result directory.
        exported_config_path = os.path.abspath(os.path.join(value_results_dir, "config.json"))
        with open(exported_config_path, "w") as f:
            json.dump(run_config, f, indent=4)

        local_stages_config = None
        if args.stages_config:
            local_stages_config = os.path.abspath(os.path.join(run_dir, "stages_config.json"))
            with open(local_stages_config, "w") as f:
                json.dump(stages_data, f, indent=4)
            exported_stages_config_path = os.path.abspath(os.path.join(value_results_dir, "stages_config.json"))
            with open(exported_stages_config_path, "w") as f:
                json.dump(stages_data, f, indent=4)

        try:
            os.chdir(run_dir)

            # 1. Generate one shared dataset per sweep value, then reuse it for all model types.
            print("\n[1/4] Generating shared train/test data...")
            run_dataset_generation(config_path, run_dir)

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
                train_argv = [
                    "--config_path", config_path,
                    "--dataset_path", dataset_path,
                    "--checkpoint_dir", model_dir,
                    "--model_type", model_type,
                    "--wandb_name_prefix", model_wandb_prefix,
                    "--wandb_project", args.wandb_project,
                ]
                if local_stages_config:
                    train_argv.extend(["--stages_config", local_stages_config])
                else:
                    train_argv.extend(["--stages_config", config_path])
                if force_doa_only:
                    train_argv.append("--train_doa_only")

                call_trainer_direct(train_argv)

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

                    test_argv = [
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
                        test_argv.append("--train_doa_only")
                    if args.log_eval_to_wandb:
                        test_argv.append("--log_to_wandb")
                    if args.visualize:
                        test_argv.append("--visualize")
                    if args.visualize_sigma:
                        test_argv.append("--visualize_sigma")

                    test_metrics, test_output = call_test_direct(test_argv)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print(test_output)
                    output_text_path = os.path.abspath(os.path.join(text_outputs_dir, f"{viz_folder_name}.txt"))
                    with open(output_text_path, "w", encoding="utf-8") as f:
                        f.write(test_output)

                    parsed_metrics = merge_metrics(test_metrics, test_output)
                    parsed_row = write_csv_row(
                        summary_csv_path,
                        args.param,
                        val,
                        model_type,
                        model_wandb_prefix,
                        viz_folder_name,
                        parsed_metrics,
                    )
                    write_csv_row(
                        value_csv_path,
                        args.param,
                        val,
                        model_type,
                        model_wandb_prefix,
                        viz_folder_name,
                        parsed_metrics,
                    )

                    record = append_full_record(
                        all_records,
                        param_name=args.param,
                        param_val=val,
                        model_type=model_type,
                        wandb_prefix=model_wandb_prefix,
                        eval_target=viz_folder_name,
                        checkpoint_path=checkpoint_to_test,
                        config_path=config_path,
                        dataset_path=dataset_path,
                        test_dataset_path=test_dataset_path,
                        loss_function=loss_func,
                        lambda_val=l_val,
                        test_metrics=test_metrics,
                        output_text=test_output,
                        parsed_row=parsed_row,
                        output_text_path=output_text_path,
                    )
                    value_records.append(record)
                    save_results_npy(all_records, summary_npy_path)
                    save_results_npy(value_records, run_results_npy_path)

            # 4. Evaluate ESPRIT once per sweep value on the same test set.
            print("\n[4/4] Evaluating ESPRIT baseline...")
            esprit_checkpoint = os.path.abspath(os.path.join(run_dir, args.model_types[0], "latest.pth"))
            esprit_argv = [
                "--config_path", config_path,
                "--test_dataset_path", test_dataset_path,
                "--checkpoint_path", esprit_checkpoint,
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
                esprit_argv.append("--visualize")
            if args.visualize_sigma:
                esprit_argv.append("--visualize_sigma")

            esprit_metrics, esprit_output = call_test_direct(esprit_argv)
            print(esprit_output)

            output_text_path = os.path.abspath(os.path.join(text_outputs_dir, "ESPRIT_Baseline.txt"))
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(esprit_output)

            parsed_metrics = merge_metrics(esprit_metrics, esprit_output)
            parsed_row = write_csv_row(
                summary_csv_path,
                args.param,
                val,
                "esprit",
                "esprit_baseline",
                "ESPRIT_Baseline",
                parsed_metrics,
            )
            write_csv_row(
                value_csv_path,
                args.param,
                val,
                "esprit",
                "esprit_baseline",
                "ESPRIT_Baseline",
                parsed_metrics,
            )

            record = append_full_record(
                all_records,
                param_name=args.param,
                param_val=val,
                model_type="esprit",
                wandb_prefix="esprit_baseline",
                eval_target="ESPRIT_Baseline",
                checkpoint_path=esprit_checkpoint,
                config_path=config_path,
                dataset_path=dataset_path,
                test_dataset_path=test_dataset_path,
                loss_function="CombinedUncertaintyLoss",
                lambda_val=1.0,
                test_metrics=esprit_metrics,
                output_text=esprit_output,
                parsed_row=parsed_row,
                output_text_path=output_text_path,
            )
            value_records.append(record)
            save_results_npy(all_records, summary_npy_path)
            save_results_npy(value_records, run_results_npy_path)

            manifest_path = os.path.abspath(os.path.join(value_results_dir, "RESULTS_MANIFEST.json"))
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sweep_parameter": args.param,
                        "sweep_value": val,
                        "working_run_dir": run_dir,
                        "dedicated_results_dir": value_results_dir,
                        "per_value_npy": run_results_npy_path,
                        "per_value_csv": value_csv_path,
                        "evaluation_text_outputs_dir": text_outputs_dir,
                        "global_summary_csv": summary_csv_path,
                        "global_summary_npy": summary_npy_path,
                        "num_records_for_value": len(value_records),
                    },
                    f,
                    indent=4,
                )
            print(f"Per-value exported results saved to: {value_results_dir}")

        finally:
            os.chdir(original_cwd)

    save_results_npy(all_records, summary_npy_path)
    print(f"\n✅ BENCHMARK SWEEP COMPLETE.")
    print(f"CSV summary saved to: {summary_csv_path}")
    print(f"Full Python-result NPY saved to: {summary_npy_path}")
    print(f"Per-value result directories saved under: {results_root_dir}")
    return all_records


if __name__ == "__main__":
    main()
