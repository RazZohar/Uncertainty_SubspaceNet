"""
Compare normal full-data SSN baselines against a dedicated CP-only Stage-1 model.

Training protocol:
  - full-data Stage-1 MC/Bayesian baseline: trained using the normal full dataset.pt;
  - CP Stage-1: a separate model trained only on the 80% CP train partition;
  - CP q_alpha,k: computed only from the disjoint 20% CP calibration partition;
  - Stage-2 proposed model: trained using the normal full dataset.pt;
  - every reported metric: evaluated on the same independent test_dataset.pt.

The CP interval is absolute residual split conformal:
    theta_hat_CP,k +/- q_alpha,k
and does not use a Stage-1 covariance estimate.
"""
from __future__ import annotations

# Reuse the carefully tested metric, covariance, MC-Dropout, and CP helpers.
# Only the orchestration differs: the CP model checkpoint is distinct from the
# normal full-data Stage-1 checkpoint.
from compare_ssn_stage1_absolute_cp_external_calibration_vs_stage2 import *  # noqa: F401,F403
from compare_ssn_stage1_absolute_cp_external_calibration_vs_stage2 import _jsonify


# Preserve references to imported helper symbols before defining this main().
_IMPORTED_MAIN = main  # type: ignore[name-defined]


def main(argv=None):  # noqa: C901
    import argparse
    import os
    import json
    import csv
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    parser = argparse.ArgumentParser(
        "Compare full-data Stage-1/Stage-2 against CP-only Stage-1 split conformal"
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument(
        "--full_stage1_checkpoint_path",
        required=True,
        help="Normal Stage-1 checkpoint trained with the full standard dataset.pt.",
    )
    parser.add_argument(
        "--cp_stage1_checkpoint_path",
        required=True,
        help="Dedicated CP-only Stage-1 checkpoint trained on the 80% CP train partition.",
    )
    parser.add_argument(
        "--stage2_checkpoint_path",
        required=True,
        help="Normal Stage-2 proposed checkpoint trained with the full standard dataset.pt.",
    )
    parser.add_argument(
        "--cp_calibration_dataset_path",
        required=True,
        help="The 20% calibration partition used only to compute q_alpha,k for the CP-only model.",
    )
    parser.add_argument(
        "--test_dataset_path",
        required=True,
        help="Independent final test dataset shared by every method.",
    )
    parser.add_argument("--cp_split_metadata_path", default=None)
    parser.add_argument("--out_dir", default="uq_fulltrain_ssn_vs_cp_stage1_vs_stage2")
    parser.add_argument("--full_stage1_name", default="SubspaceNet Stage 1 (full training data)")
    parser.add_argument("--cp_stage1_name", default="SubspaceNet Stage 1 (80% CP training data)")
    parser.add_argument("--stage2_name", default="SubspaceNet Stage 2 (full training data)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_mc", type=int, default=200)
    parser.add_argument("--stddev_spacing", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--progress_every_batches", type=int, default=1)
    parser.add_argument("--mc_progress_every", type=int, default=25)
    parser.add_argument("--include_full_stage1_regular", action="store_true")
    parser.add_argument("--train_doa_only", action="store_true", default=True)
    parser.add_argument("--wandb_name_prefix", type=str, default=None)
    args = parser.parse_args(argv)

    if args.alpha is None:
        args.alpha = gaussian_alpha_from_stddev_spacing(args.stddev_spacing)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress(f"Device: {device}")
    progress(f"Full-data Stage-1 checkpoint: {args.full_stage1_checkpoint_path}")
    progress(f"CP-only Stage-1 checkpoint: {args.cp_stage1_checkpoint_path}")
    progress(f"Full-data Stage-2 checkpoint: {args.stage2_checkpoint_path}")
    progress(
        f"Gaussian spacing: +/-{args.stddev_spacing:g} sigma, "
        f"target coverage={1.0 - args.alpha:.9f}, alpha={args.alpha:.9f}"
    )

    def _enable_graph_features(dataset) -> None:
        base = dataset
        while isinstance(base, Subset):
            base = base.dataset
        if hasattr(base, "set_use_graph_features"):
            base.set_use_graph_features(use_features=True)

    cp_cal_ds = torch.load(args.cp_calibration_dataset_path, weights_only=False)
    test_ds = torch.load(args.test_dataset_path, weights_only=False)
    _enable_graph_features(cp_cal_ds)
    _enable_graph_features(test_ds)
    n_cal, n_test = len(cp_cal_ds), len(test_ds)
    if n_cal < 1 or n_test < 1:
        raise ValueError("Both CP calibration and independent test datasets must contain at least one scene.")

    cal_loader = DataLoader(
        cp_cal_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=graph_scene_collate, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=graph_scene_collate, drop_last=False,
    )

    # The three models are intentionally distinct objects and checkpoints.
    full_stage1_model = load_ssn_model_from_checkpoint(
        args, test_ds, args.full_stage1_checkpoint_path, device, args.full_stage1_name
    )
    cp_stage1_model = load_ssn_model_from_checkpoint(
        args, test_ds, args.cp_stage1_checkpoint_path, device, args.cp_stage1_name
    )
    stage2_model = load_ssn_model_from_checkpoint(
        args, test_ds, args.stage2_checkpoint_path, device, args.stage2_name
    )

    progress("[1/5] CP-only Stage-1 deterministic DoA on CP calibration data...")
    cp_stage1_cal = deterministic_predictions(
        cp_stage1_model, cal_loader, device, eps=args.eps,
        label="[1/5] CP-only Stage-1 calibration inference",
        progress_every_batches=args.progress_every_batches,
    )

    progress("[2/5] CP-only Stage-1 deterministic DoA on final test data...")
    cp_stage1_test = deterministic_predictions(
        cp_stage1_model, test_loader, device, eps=args.eps,
        label="[2/5] CP-only Stage-1 test inference",
        progress_every_batches=args.progress_every_batches,
    )

    progress(f"[3/5] Full-data Stage-1 MC-Dropout on final test data, n_mc={args.n_mc}...")
    full_stage1_mc_sets = mc_dropout_prediction_sets(
        full_stage1_model, test_loader, device, n_mc=args.n_mc, eps=args.eps,
        label="[3/5] Full-data Stage-1 MC-Dropout",
        mc_progress_every=args.mc_progress_every,
    )
    full_stage1_mc_diag = diagonalize_prediction_covariance(full_stage1_mc_sets.epistemic, eps=args.eps)
    full_stage1_bayesian_diag = diagonalize_prediction_covariance(full_stage1_mc_sets.epistemic, eps=args.eps)

    progress("[4/5] Absolute split CP: q from the CP-only model and its disjoint calibration split...")
    cp_test, qhat_rad = absolute_conformalize_sourcewise(
        cp_stage1_cal, cp_stage1_test, alpha=args.alpha, eps=args.eps
    )

    progress("[5/5] Full-data Stage-2 proposed covariance on final test data...")
    stage2_test = deterministic_predictions(
        stage2_model, test_loader, device, eps=args.eps,
        label="[5/5] Full-data Stage-2 test inference",
        progress_every_batches=args.progress_every_batches,
    )

    full_regular_name = f"{args.full_stage1_name} (regular)"
    mc_name = f"{args.full_stage1_name} + MC Dropout"
    bayes_name = f"{args.full_stage1_name} + Bayesian MC Dropout"
    cp_name = f"{args.cp_stage1_name} + absolute split conformal prediction"
    proposed_name = f"{args.stage2_name} (proposed)"

    rows = []
    tensors = {
        mc_name: full_stage1_mc_diag,
        bayes_name: full_stage1_bayesian_diag,
        cp_name: cp_test,
        proposed_name: stage2_test,
    }

    if args.include_full_stage1_regular:
        progress("Computing optional full-data deterministic Stage-1 reference row...")
        full_stage1_test = deterministic_predictions(
            full_stage1_model, test_loader, device, eps=args.eps,
            label="Optional full-data Stage-1 deterministic inference",
            progress_every_batches=args.progress_every_batches,
        )
        row = covariance_metrics(
            full_regular_name, full_stage1_test, alpha=args.alpha,
            stddev_spacing=args.stddev_spacing, eps=args.eps,
        )
        row.update({
            "checkpoint_role": "full_data_stage1_reference",
            "training_data_protocol": "normal full dataset.pt training",
            "covariance_definition": "Full-data Stage-1 deterministic model covariance",
        })
        rows.append(row)
        tensors[full_regular_name] = full_stage1_test

    mc_row = covariance_metrics(
        mc_name, full_stage1_mc_diag, alpha=args.alpha,
        stddev_spacing=args.stddev_spacing, eps=args.eps,
    )
    mc_row.update({
        "checkpoint_role": "full_data_stage1",
        "training_data_protocol": "normal full dataset.pt training",
        "covariance_definition": "diag(Var_w[theta_hat_w]) from full-data Stage-1 MC-Dropout",
        "n_mc": args.n_mc,
        "diagonal_covariance_only": True,
    })
    rows.append(mc_row)

    bayes_row = covariance_metrics(
        bayes_name, full_stage1_bayesian_diag, alpha=args.alpha,
        stddev_spacing=args.stddev_spacing, eps=args.eps,
    )
    bayes_row.update({
        "checkpoint_role": "full_data_stage1",
        "training_data_protocol": "normal full dataset.pt training",
        "covariance_definition": "diag(Var_w[theta_hat_w]) from full-data Stage-1 stochastic DoA outputs",
        "n_mc": args.n_mc,
        "diagonal_covariance_only": True,
    })
    rows.append(bayes_row)

    cp_row = covariance_metrics(
        cp_name, cp_test, alpha=args.alpha,
        stddev_spacing=args.stddev_spacing, eps=args.eps,
    )
    cp_row.update({
        "checkpoint_role": "cp_only_stage1",
        "training_data_protocol": "dedicated CP Stage-1: 80% CP train partition only",
        "conformal_mode": "per_source_absolute_residual",
        "conformal_base": "CP-only Stage-1 deterministic DoA; no model covariance used",
        "covariance_definition": "diag(q_alpha,k^2) diagnostic representation of absolute CP intervals",
        "qhat_rad": _jsonify(qhat_rad.cpu()),
        "qhat_deg": _jsonify(torch.rad2deg(qhat_rad).cpu()),
        "cp_interval_definition": "theta_hat_(ordered,k) +/- q_alpha,k",
        "cp_source_association": "ascending DoA order; no ground-truth matching used to define CP source index",
        "cp_calibration_num_scenes": int(n_cal),
        "final_evaluation_num_scenes": int(n_test),
        "cp_calibration_dataset_path": os.path.abspath(args.cp_calibration_dataset_path),
        "full_stage1_checkpoint_path": os.path.abspath(args.full_stage1_checkpoint_path),
        "cp_stage1_checkpoint_path": os.path.abspath(args.cp_stage1_checkpoint_path),
        "test_dataset_path": os.path.abspath(args.test_dataset_path),
        "joint_metrics_note": (
            "Diagnostic only: absolute source-wise CP guarantees marginal coverage, "
            "not chi-square joint-region coverage."
        ),
    })
    rows.append(cp_row)

    proposed_row = covariance_metrics(
        proposed_name, stage2_test, alpha=args.alpha,
        stddev_spacing=args.stddev_spacing, eps=args.eps,
    )
    proposed_row.update({
        "checkpoint_role": "full_data_stage2_proposed",
        "training_data_protocol": "normal full dataset.pt training",
        "covariance_definition": "Full-data Stage-2 proposed full covariance",
    })
    rows.append(proposed_row)

    for row in rows:
        row.setdefault("cp_calibration_num_scenes", int(n_cal))
        row.setdefault("final_evaluation_num_scenes", int(n_test))
        row.setdefault("test_dataset_path", os.path.abspath(args.test_dataset_path))
        row["reported_on_full_independent_test_dataset"] = True

    print_table(rows)
    progress("Metric table computed; writing files.")
    os.makedirs(args.out_dir, exist_ok=True)

    split_metadata = {
        "full_stage1_checkpoint_path": os.path.abspath(args.full_stage1_checkpoint_path),
        "cp_stage1_checkpoint_path": os.path.abspath(args.cp_stage1_checkpoint_path),
        "stage2_checkpoint_path": os.path.abspath(args.stage2_checkpoint_path),
        "cp_calibration_dataset_path": os.path.abspath(args.cp_calibration_dataset_path),
        "test_dataset_path": os.path.abspath(args.test_dataset_path),
        "cp_calibration_num_scenes": int(n_cal),
        "final_evaluation_num_scenes": int(n_test),
        "cp_split_metadata_path": os.path.abspath(args.cp_split_metadata_path) if args.cp_split_metadata_path else None,
        "protocol": (
            "MC/Bayesian uses full-data Stage-1; CP uses distinct 80%-trained Stage-1 and its "
            "disjoint 20% calibration partition; Stage-2 uses full-data checkpoint."
        ),
    }
    if args.cp_split_metadata_path:
        with open(args.cp_split_metadata_path, "r", encoding="utf-8") as f:
            split_metadata["cp_training_split_metadata"] = json.load(f)
    with open(os.path.join(args.out_dir, "fulltrain_vs_cp_stage1_data_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonify(split_metadata), f, indent=2)

    # Reuse existing generic writer for standard CSV/JSON/PT filenames.
    write_outputs(args.out_dir, rows, tensors)

    csv_path = os.path.join(args.out_dir, "fulltrain_ssn_vs_cp_stage1_vs_stage2_comparison.csv")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.out_dir, "fulltrain_ssn_vs_cp_stage1_vs_stage2_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(rows), f, indent=2)

    progress(f"Comparison files written: {csv_path} | {json_path}")
    return rows


if __name__ == "__main__":
    main()
