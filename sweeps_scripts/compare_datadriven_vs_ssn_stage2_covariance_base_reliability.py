#!/usr/bin/env python3
"""
compare_datadriven_vs_ssn_stage2_covariance.py

Standalone covariance-UQ comparison script for your current pipeline.
It does NOT modify trainer.py, test.py, or run_sweep.py.

It compares exactly these four covariance-producing methods:

  1. data_driven_regular
     DataDrivenComplexNet deterministic output covariance_i.

  2. data_driven_mc_dropout
     Bayesian / MC-Dropout covariance from stochastic DataDrivenComplexNet
     bearing predictions with dropout kept active during inference.

  3. data_driven_conformal_prediction
     Source-wise conformal calibration of DataDrivenComplexNet deterministic
     covariance.  The conformal alpha is derived by default from a Gaussian
     two-sided ±1 std-dev interval:
         coverage = P(|Z| <= 1), Z~N(0,1) = 0.6826894921370859
         alpha    = 1 - coverage = 0.31731050786291415

     Per-source conformal scores:
         score[i,s,k] = |theta_true[i,s,k] - mu[i,s,k]| / sqrt(Sigma[i,s,k,k])

     Per-source qhat is applied to the full covariance as:
         D = diag(qhat_1, ..., qhat_P)
         Sigma_CP = D @ Sigma @ D

  4. SSN Stage 2
     MultiSubarraysModel deterministic output covariance_i.

Outputs:
  <out_dir>/covariance_method_comparison.csv
  <out_dir>/covariance_method_comparison.json
  <out_dir>/covariance_method_tensors.pt

Example:
python compare_datadriven_vs_ssn_stage2_covariance.py \
  --config_path config.json \
  --test_dataset_path test_dataset.pt \
  --cal_dataset_path dataset.pt \
  --data_driven_checkpoint_path data_driven_complex/latest.pth \
  --multi_subarray_checkpoint_path multi_subarray/best_base_reliability.pth \
  --batch_size 1024 \
  --n_mc 200 \
  --stddev_spacing 1.0 \
  --out_dir uq_cov_comparison
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Intended to be copied into the project root, next to trainer.py/test.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer import graph_scene_collate, infer_dataset_dimensions
from src.models import DataDrivenComplexNet
from src.multi_subarrays_model import MultiSubarraysModel
from src.multi_model_dataset import SensorSourceGraphDataset

try:
    from src.criterions import model_output_to_covariance_rad2
except Exception:
    model_output_to_covariance_rad2 = None


GAUSSIAN_1STD_COVERAGE = 0.6826894921370859
GAUSSIAN_1STD_ALPHA = 1.0 - GAUSSIAN_1STD_COVERAGE


@dataclass
class PredictionSet:
    mean_rad: torch.Tensor      # [N, S, P]
    cov_rad2: torch.Tensor      # [N, S, P, P]
    target_rad: torch.Tensor    # [N, S, P]


# -----------------------------------------------------------------------------
# Numeric utilities
# -----------------------------------------------------------------------------

def gaussian_two_sided_coverage(stddev_spacing: float) -> float:
    """P(|Z| <= stddev_spacing) for Z ~ N(0,1)."""
    return float(math.erf(float(stddev_spacing) / math.sqrt(2.0)))


def gaussian_alpha_from_stddev_spacing(stddev_spacing: float) -> float:
    """Error rate alpha corresponding to a Gaussian two-sided ±kσ interval."""
    return 1.0 - gaussian_two_sided_coverage(stddev_spacing)


def _jsonify(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


def wrap_periodic_error(x: torch.Tensor, period: float = math.pi) -> torch.Tensor:
    """Wrap ULA DOA error to [-period/2, period/2). Use period=pi for DOA ambiguity."""
    return ((x + period / 2.0) % period) - period / 2.0


def make_spd(cov: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Symmetrize covariance and add adaptive diagonal jitter."""
    cov = cov.float()
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    p = cov.shape[-1]
    eye = torch.eye(p, dtype=cov.dtype, device=cov.device)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1, keepdim=True).clamp_min(eps)
    jitter = (eps + 1e-6 * trace / max(p, 1)).unsqueeze(-1) * eye
    return cov + jitter


def chi2_ppf_approx(prob: float, df: int) -> float:
    """Chi-square inverse CDF. Uses scipy if available, otherwise Wilson-Hilferty."""
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(prob, df))
    except Exception:
        try:
            from statistics import NormalDist
            z = NormalDist().inv_cdf(prob)
        except Exception:
            # Good fallback for prob≈0.682689.
            z = 0.475232
        return float(df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3)


def conformal_quantile(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Split-conformal quantile with finite-sample correction:
        qhat = kth smallest score, k = ceil((n+1)(1-alpha)).
    """
    s = scores.detach().flatten()
    s = s[torch.isfinite(s)]
    if s.numel() == 0:
        raise ValueError("No finite conformal scores were available.")
    n = s.numel()
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return torch.sort(s).values[k - 1]


# -----------------------------------------------------------------------------
# Source alignment and covariance extraction
# -----------------------------------------------------------------------------

def sorted_by_doa(
    mean_rad: torch.Tensor,
    cov_rad2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sort predicted sources by DOA and reorder covariance rows/cols consistently."""
    sorted_mean, order = torch.sort(mean_rad, dim=-1)
    if cov_rad2 is None:
        return sorted_mean, None
    p = cov_rad2.shape[-1]
    row_idx = order.unsqueeze(-1).expand(*order.shape, p)
    cov_sorted = torch.gather(cov_rad2, dim=-2, index=row_idx)
    col_idx = order.unsqueeze(-2).expand(*order.shape[:-1], p, p)
    cov_sorted = torch.gather(cov_sorted, dim=-1, index=col_idx)
    return sorted_mean, cov_sorted


def align_to_target(
    mean_rad: torch.Tensor,
    target_rad: torch.Tensor,
    cov_rad2: Optional[torch.Tensor] = None,
    period: float = math.pi,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Find best source permutation per [batch, subarray] using target DOAs.
    Returns aligned mean, aligned covariance, and aligned error.
    """
    if mean_rad.shape != target_rad.shape:
        raise ValueError(f"mean and target shapes must match, got {mean_rad.shape} and {target_rad.shape}")

    b, s, p = mean_rad.shape
    perms = torch.tensor(list(itertools.permutations(range(p))), device=mean_rad.device, dtype=torch.long)
    nperm = perms.shape[0]

    mean_exp = mean_rad.unsqueeze(2).expand(b, s, nperm, p)
    perm_exp = perms.view(1, 1, nperm, p).expand(b, s, nperm, p)
    mean_perm = torch.gather(mean_exp, dim=3, index=perm_exp)

    err_perm = wrap_periodic_error(mean_perm - target_rad.unsqueeze(2), period=period)
    cost = torch.sum(err_perm ** 2, dim=-1)
    best_idx = torch.argmin(cost, dim=2)  # [B,S]

    gather_idx = best_idx.view(b, s, 1, 1).expand(b, s, 1, p)
    mean_best = torch.gather(mean_perm, dim=2, index=gather_idx).squeeze(2)
    err_best = torch.gather(err_perm, dim=2, index=gather_idx).squeeze(2)

    cov_best = None
    if cov_rad2 is not None:
        cov_exp = cov_rad2.unsqueeze(2).expand(b, s, nperm, p, p)
        row_idx = perms.view(1, 1, nperm, p, 1).expand(b, s, nperm, p, p)
        cov_rows = torch.gather(cov_exp, dim=3, index=row_idx)
        col_idx = perms.view(1, 1, nperm, 1, p).expand(b, s, nperm, p, p)
        cov_perm = torch.gather(cov_rows, dim=4, index=col_idx)
        cov_best = torch.gather(
            cov_perm,
            dim=2,
            index=best_idx.view(b, s, 1, 1, 1).expand(b, s, 1, p, p),
        ).squeeze(2)

    return mean_best, cov_best, err_best


def extract_covariance_rad2(model_result: Dict[str, torch.Tensor], eps: float = 1e-10) -> torch.Tensor:
    """Extract covariance in radians^2 from a model output dictionary."""
    if "covariance_i" in model_result and model_result["covariance_i"] is not None:
        return make_spd(model_result["covariance_i"].float(), eps=eps)

    if model_output_to_covariance_rad2 is not None:
        try:
            return make_spd(model_output_to_covariance_rad2(model_result).float(), eps=eps)
        except Exception:
            pass

    if "cov_i" in model_result and model_result["cov_i"] is not None:
        # MultiSubarraysModel debug convention: cov_i is deg^2.
        return make_spd(model_result["cov_i"].float() * ((math.pi / 180.0) ** 2), eps=eps)

    if "sigma_i" not in model_result:
        raise KeyError("Model output does not contain covariance_i, cov_i, or sigma_i.")

    # DataDrivenComplexNet convention: sigma_i is in degrees.
    sigma_rad = torch.deg2rad(model_result["sigma_i"].float().clamp_min(eps))
    return make_spd(torch.diag_embed(sigma_rad ** 2), eps=eps)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_data_driven_model(
    args: argparse.Namespace,
    dataset: SensorSourceGraphDataset,
    device: torch.device,
) -> nn.Module:
    num_antennas, num_snapshots, num_sources = infer_dataset_dimensions(dataset)
    model = DataDrivenComplexNet(
        N=num_antennas,
        T=num_snapshots,
        tau=args.tau,
        M=num_sources,
    ).to(device)
    model.estimate_uncertainty = True
    model.estimate_position = False

    checkpoint = torch.load(args.data_driven_checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def load_multi_subarray_model(
    args: argparse.Namespace,
    dataset: SensorSourceGraphDataset,
    device: torch.device,
) -> nn.Module:
    sensor_positions = dataset.get_sensor_potision()
    model = MultiSubarraysModel(
        sensors_positions=sensor_positions,
        multi_model_configuration=args.config_path,
        args=args,
    ).to(device)
    model.estimate_uncertainty = True
    model.estimate_position = False
    if hasattr(model, "enable_full_covariance_estimation"):
        model.enable_full_covariance_estimation()

    checkpoint = torch.load(args.multi_subarray_checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Prediction collection
# -----------------------------------------------------------------------------

@torch.no_grad()
def deterministic_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float = 1e-10,
) -> PredictionSet:
    model.eval()
    means: List[torch.Tensor] = []
    covs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for sensor_positions, _source_positions, samples, doa_gt in loader:
        sensor_positions = sensor_positions.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        doa_gt = torch.deg2rad(doa_gt.to(device, non_blocking=True)).float()

        out = model(sensor_positions, samples, doa_gt)
        if not isinstance(out, dict) or "bearings" not in out:
            raise ValueError("Expected model output dict with key 'bearings'.")

        mean = out["bearings"].float()
        cov = extract_covariance_rad2(out, eps=eps)
        mean, cov, _ = align_to_target(mean, doa_gt, cov)

        means.append(mean.detach().cpu())
        covs.append(cov.detach().cpu())
        targets.append(doa_gt.detach().cpu())

    return PredictionSet(
        mean_rad=torch.cat(means, dim=0),
        cov_rad2=torch.cat(covs, dim=0),
        target_rad=torch.cat(targets, dim=0),
    )


def enable_dropout_only(model: nn.Module) -> None:
    """Keep BatchNorm/etc in eval mode, but activate dropout modules for MC-Dropout."""
    model.eval()
    for module in model.modules():
        name = module.__class__.__name__.lower()
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)) or "dropout" in name:
            module.train()


@torch.no_grad()
def mc_dropout_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_mc: int,
    eps: float = 1e-10,
) -> PredictionSet:
    """
    Bayesian / MC-Dropout covariance from stochastic DOA predictions.
    The covariance is the sample covariance of the MC bearing outputs.
    """
    means: List[torch.Tensor] = []
    covs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for sensor_positions, _source_positions, samples, doa_gt in loader:
        sensor_positions = sensor_positions.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        doa_gt = torch.deg2rad(doa_gt.to(device, non_blocking=True)).float()

        preds_t: List[torch.Tensor] = []
        for _ in range(n_mc):
            enable_dropout_only(model)
            out = model(sensor_positions, samples, doa_gt)
            pred = out["bearings"].float()
            # Sort without GT before computing MC covariance so component order is stable.
            pred, _ = sorted_by_doa(pred, None)
            preds_t.append(pred)

        model.eval()
        pred_stack = torch.stack(preds_t, dim=0)  # [T,B,S,P]
        mu = pred_stack.mean(dim=0)
        residual = wrap_periodic_error(pred_stack - mu.unsqueeze(0), period=math.pi)

        if n_mc > 1:
            cov_mc = torch.einsum("tbsi,tbsj->bsij", residual, residual) / float(n_mc - 1)
        else:
            p = mu.shape[-1]
            cov_mc = torch.zeros(*mu.shape, p, device=device, dtype=mu.dtype)

        cov_mc = make_spd(cov_mc, eps=eps)
        # Align to GT only after the MC covariance is formed.
        mu, cov_mc, _ = align_to_target(mu, doa_gt, cov_mc)

        means.append(mu.detach().cpu())
        covs.append(cov_mc.detach().cpu())
        targets.append(doa_gt.detach().cpu())

    return PredictionSet(
        mean_rad=torch.cat(means, dim=0),
        cov_rad2=torch.cat(covs, dim=0),
        target_rad=torch.cat(targets, dim=0),
    )


# -----------------------------------------------------------------------------
# Source-wise conformal prediction for covariance
# -----------------------------------------------------------------------------

def sourcewise_normalized_scores(
    pred: PredictionSet,
    eps: float = 1e-10,
    period: float = math.pi,
) -> torch.Tensor:
    """scores [N,S,P] = |error_k| / sqrt(Sigma_kk), after source matching."""
    _, cov_aligned, err = align_to_target(pred.mean_rad, pred.target_rad, pred.cov_rad2, period=period)
    cov_aligned = make_spd(cov_aligned, eps=eps)
    sigma = torch.sqrt(torch.diagonal(cov_aligned, dim1=-2, dim2=-1).clamp_min(eps))
    return torch.abs(err) / sigma


def conformal_quantile_by_source(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return qhat[P], aggregating over all calibration samples and subarrays per source."""
    qhats = []
    for k in range(scores.shape[-1]):
        qhats.append(conformal_quantile(scores[..., k], alpha=alpha))
    return torch.stack(qhats, dim=0)  # [P]


def conformalize_sourcewise(
    cal_pred: PredictionSet,
    test_pred: PredictionSet,
    alpha: float,
    eps: float = 1e-10,
) -> Tuple[PredictionSet, torch.Tensor]:
    """
    Source-wise CP covariance calibration:
        score_k = |e_k| / sqrt(Sigma_kk)
        Sigma_CP = D Sigma D, D=diag(qhat_1,...,qhat_P)
    """
    scores = sourcewise_normalized_scores(cal_pred, eps=eps)
    qhat = conformal_quantile_by_source(scores, alpha=alpha).float()

    q = qhat.view(1, 1, -1).to(test_pred.cov_rad2.device)
    cov_cp = test_pred.cov_rad2.float() * q.unsqueeze(-1) * q.unsqueeze(-2)
    cov_cp = make_spd(cov_cp, eps=eps)
    return PredictionSet(test_pred.mean_rad, cov_cp, test_pred.target_rad), qhat


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def covariance_metrics(
    name: str,
    pred: PredictionSet,
    alpha: float,
    stddev_spacing: float,
    eps: float = 1e-10,
    period: float = math.pi,
) -> Dict[str, Any]:
    mean, cov, err = align_to_target(pred.mean_rad, pred.target_rad, pred.cov_rad2, period=period)
    cov = make_spd(cov, eps=eps)

    n, s, p = mean.shape
    flat_e = err.reshape(-1, p).float()
    flat_cov = cov.reshape(-1, p, p).float()
    f = flat_e.shape[0]

    sol = torch.linalg.solve(flat_cov, flat_e.unsqueeze(-1)).squeeze(-1)
    nees = torch.sum(flat_e * sol, dim=-1).clamp_min(0.0)
    anees_raw = nees.mean()
    anees_norm = anees_raw / float(p)

    rmse_rad = torch.sqrt(torch.mean(flat_e ** 2))
    mae_rad = torch.mean(torch.abs(flat_e))

    target_coverage = 1.0 - alpha
    chi2_thr = chi2_ppf_approx(target_coverage, p)
    coverage_chi2_target = (nees <= chi2_thr).float().mean()

    diag = torch.diagonal(flat_cov, dim1=-2, dim2=-1).clamp_min(eps)
    sigma = torch.sqrt(diag)
    sigma_deg = torch.rad2deg(sigma)

    # Marginal source-wise coverage for the requested Gaussian ±kσ spacing.
    source_hit = (torch.abs(flat_e) <= float(stddev_spacing) * sigma).float()
    source_coverage_target = source_hit.mean()
    source_coverage_per_source = source_hit.reshape(n, s, p).mean(dim=(0, 1))

    # APEC/EEC consistency.
    apec = flat_cov.mean(dim=0)
    eec = (flat_e.T @ flat_e) / float(max(f, 1))
    apec_trace = torch.trace(apec)
    eec_trace = torch.trace(eec)
    trace_gap = torch.abs(apec_trace - eec_trace)
    fro_gap = torch.linalg.norm(apec - eec, ord="fro")
    eec_fro = torch.linalg.norm(eec, ord="fro").clamp_min(eps)
    fro_rel = fro_gap / eec_fro

    sign, logabsdet = torch.linalg.slogdet(flat_cov)
    nll = 0.5 * (nees + logabsdet + p * math.log(2.0 * math.pi))
    nll = nll[torch.isfinite(nll)].mean()
    valid_logdet = logabsdet[sign > 0]
    mean_logdet = valid_logdet.mean() if valid_logdet.numel() else torch.tensor(float("nan"))

    return {
        "method": name,
        "num_vectors": int(f),
        "num_sources": int(p),
        "alpha": float(alpha),
        "target_coverage": float(target_coverage),
        "stddev_spacing": float(stddev_spacing),
        "rmse_deg": float(torch.rad2deg(rmse_rad).cpu()),
        "mae_deg": float(torch.rad2deg(mae_rad).cpu()),
        "anees_raw": float(anees_raw.cpu()),
        "anees_normalized": float(anees_norm.cpu()),
        "log_anees_normalized": float(torch.log(anees_norm.clamp_min(eps)).cpu()),
        "coverage_chi2_target": float(coverage_chi2_target.cpu()),
        "coverage_chi2_target_error": float(torch.abs(coverage_chi2_target - target_coverage).cpu()),
        "chi2_target_threshold": float(chi2_thr),
        "source_coverage_target": float(source_coverage_target.cpu()),
        "source_coverage_target_error": float(torch.abs(source_coverage_target - target_coverage).cpu()),
        "source_coverage_per_source": _jsonify(source_coverage_per_source.cpu()),
        "mean_sigma_deg": float(sigma_deg.mean().cpu()),
        "median_sigma_deg": float(sigma_deg.median().cpu()),
        "mean_trace_cov_rad2": float(apec_trace.cpu()),
        "mean_trace_cov_deg2": float((apec_trace * (180.0 / math.pi) ** 2).cpu()),
        "mean_logdet_cov": float(mean_logdet.cpu()),
        "gaussian_nll": float(nll.cpu()),
        "apec_trace": float(apec_trace.cpu()),
        "eec_trace": float(eec_trace.cpu()),
        "apec_eec_trace_gap": float(trace_gap.cpu()),
        "apec_eec_fro": float(fro_gap.cpu()),
        "apec_eec_rel": float(fro_rel.cpu()),
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    keys = [
        "method",
        "rmse_deg",
        "source_coverage_target",
        "source_coverage_target_error",
        "anees_normalized",
        "mean_sigma_deg",
        "mean_trace_cov_deg2",
        "apec_eec_rel",
        "qhat",
    ]
    widths = {k: max(len(k), 12) for k in keys}
    for r in rows:
        for k in keys:
            val = r.get(k, "")
            text = f"{val:.6g}" if isinstance(val, float) else str(val)
            widths[k] = max(widths[k], len(text))

    print("\n" + " | ".join(k.ljust(widths[k]) for k in keys))
    print("-+-".join("-" * widths[k] for k in keys))
    for r in rows:
        cells = []
        for k in keys:
            val = r.get(k, "")
            text = f"{val:.6g}" if isinstance(val, float) else str(val)
            cells.append(text.ljust(widths[k]))
        print(" | ".join(cells))


def write_outputs(out_dir: str, rows: List[Dict[str, Any]], tensors: Dict[str, PredictionSet]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "covariance_method_comparison.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    json_path = os.path.join(out_dir, "covariance_method_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(rows), f, indent=2)

    pt_path = os.path.join(out_dir, "covariance_method_tensors.pt")
    torch.save({
        k: {
            "mean_rad": v.mean_rad,
            "cov_rad2": v.cov_rad2,
            "target_rad": v.target_rad,
        }
        for k, v in tensors.items()
    }, pt_path)

    print(f"\nSaved:\n  {csv_path}\n  {json_path}\n  {pt_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    parser = argparse.ArgumentParser(
        "Compare DataDrivenComplexNet regular / MC-Dropout / CP covariance against SSN Stage 2 covariance"
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--test_dataset_path", required=True)
    parser.add_argument("--cal_dataset_path", default=None,
                        help="Separate calibration dataset for DataDriven conformal prediction. If omitted, a prefix of the test set is used.")
    parser.add_argument("--data_driven_checkpoint_path", required=True)
    parser.add_argument("--multi_subarray_checkpoint_path", required=True)
    parser.add_argument("--multi_subarray_method_name", default="SSN Stage 2",
                        help="Display name for the multi-subarray/SSN row. Default: SSN Stage 2")
    parser.add_argument("--out_dir", default="uq_covariance_method_comparison")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--n_mc", type=int, default=200)
    parser.add_argument("--stddev_spacing", type=float, default=1.0,
                        help="Gaussian ±k std-dev spacing used to choose alpha. Default k=1 gives 68.2689% coverage.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Optional CP error rate. If omitted, computed from --stddev_spacing using Gaussian two-sided coverage.")
    parser.add_argument("--cal_split", type=float, default=0.1,
                        help="Used only when --cal_dataset_path is omitted; first fraction of test set is CP calibration.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-10)

    # Compatibility attributes expected by MultiSubarraysModel.
    parser.add_argument("--train_doa_only", action="store_true", default=True)
    parser.add_argument("--wandb_name_prefix", type=str, default=None)

    args = parser.parse_args(argv)

    if args.alpha is None:
        args.alpha = gaussian_alpha_from_stddev_spacing(args.stddev_spacing)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"stddev_spacing={args.stddev_spacing:.6f}")
    print(f"alpha={args.alpha:.12f}, target coverage={1.0 - args.alpha:.6f}")
    print(f"n_mc={args.n_mc}")

    test_ds: SensorSourceGraphDataset = torch.load(args.test_dataset_path, weights_only=False)
    test_ds.set_use_graph_features(use_features=True)

    if args.cal_dataset_path:
        cal_ds: SensorSourceGraphDataset = torch.load(args.cal_dataset_path, weights_only=False)
        cal_ds.set_use_graph_features(use_features=True)
        eval_ds = test_ds
    else:
        n_total = len(test_ds)
        n_cal = max(1, int(round(n_total * args.cal_split)))
        if n_cal >= n_total:
            raise ValueError("cal_split leaves no test samples. Use smaller --cal_split or provide --cal_dataset_path.")
        cal_ds = Subset(test_ds, list(range(n_cal)))
        eval_ds = Subset(test_ds, list(range(n_cal, n_total)))
        print(f"No cal_dataset_path provided: using first {n_cal}/{n_total} test samples for CP calibration.")

    cal_loader = DataLoader(
        cal_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_scene_collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=graph_scene_collate,
        drop_last=False,
    )

    print("\nBuilding models...")
    data_driven = load_data_driven_model(args, test_ds, device)
    multi_subarray = load_multi_subarray_model(args, test_ds, device)

    print("\n[1/5] DataDrivenComplexNet deterministic covariance on calibration set...")
    dd_cal = deterministic_predictions(data_driven, cal_loader, device, eps=args.eps)

    print("[2/5] DataDrivenComplexNet deterministic covariance on test set...")
    dd_test = deterministic_predictions(data_driven, test_loader, device, eps=args.eps)

    print(f"[3/5] DataDrivenComplexNet MC-Dropout covariance on test set, n_mc={args.n_mc}...")
    dd_mc_test = mc_dropout_predictions(data_driven, test_loader, device, n_mc=args.n_mc, eps=args.eps)

    print("[4/5] DataDrivenComplexNet source-wise conformal covariance calibration...")
    dd_cp_test, qhat = conformalize_sourcewise(dd_cal, dd_test, alpha=args.alpha, eps=args.eps)

    print(f"[5/5] {args.multi_subarray_method_name} deterministic covariance on test set...")
    ms_test = deterministic_predictions(multi_subarray, test_loader, device, eps=args.eps)

    rows: List[Dict[str, Any]] = []
    tensors: Dict[str, PredictionSet] = {}

    tensors["data_driven_regular"] = dd_test
    rows.append(covariance_metrics(
        "data_driven_regular",
        dd_test,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    ))

    tensors["data_driven_mc_dropout"] = dd_mc_test
    rows.append(covariance_metrics(
        "data_driven_mc_dropout",
        dd_mc_test,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    ))

    tensors["data_driven_conformal_prediction"] = dd_cp_test
    cp_row = covariance_metrics(
        "data_driven_conformal_prediction",
        dd_cp_test,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    )
    cp_row["qhat"] = _jsonify(qhat.cpu())
    cp_row["base_method"] = "data_driven_regular"
    cp_row["conformal_mode"] = "per_source"
    rows.append(cp_row)

    tensors[args.multi_subarray_method_name] = ms_test
    rows.append(covariance_metrics(
        args.multi_subarray_method_name,
        ms_test,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    ))

    print_table(rows)
    write_outputs(args.out_dir, rows, tensors)
    return rows


if __name__ == "__main__":
    main()
