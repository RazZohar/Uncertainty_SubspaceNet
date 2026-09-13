"""
compare_ssn_stage1_absolute_cp_vs_stage2.py

Compare Stage-1 uncertainty baselines with the Stage-2 proposed covariance.

Stage-1 / best_base_accuracy.pth is the base estimator for all baselines:
  1) MC-Dropout: MC mean DoA + diagonal variance of stochastic DoA outputs.
  2) Bayesian MC-Dropout: the same diagonal posterior-predictive MC variance.
     Under this requested diagonal-only definition, rows (1) and (2) are
     numerically identical and are retained only as separately named baselines.
  3) Absolute-residual source-wise split conformal prediction:
     - reserve a random 20% of the originally independent test_dataset.pt as
       D_cal,CP;
     - use the remaining 80% as D_eval;
     - estimate one absolute half-width q_{alpha,k} per source from D_cal,CP;
     - return [theta_hat_k - q_{alpha,k}, theta_hat_k + q_{alpha,k}].
     This conformal baseline DOES NOT use the Stage-1 covariance matrix.
     Its comparison covariance is diag(q_{alpha,1}^2,...,q_{alpha,M}^2).

Stage-2 / best_base_reliability.pth is the proposed solution and supplies its
full predicted covariance matrix directly.

No training and no dataset generation are performed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer import graph_scene_collate
from src.multi_subarrays_model import MultiSubarraysModel
from src.multi_model_dataset import SensorSourceGraphDataset

try:
    from src.criterions import model_output_to_covariance_rad2
except Exception:
    model_output_to_covariance_rad2 = None


GAUSSIAN_1STD_COVERAGE = 0.6826894921370859
GAUSSIAN_1STD_ALPHA = 1.0 - GAUSSIAN_1STD_COVERAGE


def progress(message: str) -> None:
    """Emit an unbuffered, timestamped heartbeat for Slurm log monitoring."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass
class PredictionSet:
    mean_rad: torch.Tensor
    cov_rad2: torch.Tensor
    target_rad: torch.Tensor


@dataclass
class MCDropoutPredictionSets:
    epistemic: PredictionSet
    mean_aleatoric: PredictionSet
    bayesian_total: PredictionSet


# -----------------------------------------------------------------------------
# Numeric utilities
# -----------------------------------------------------------------------------

def gaussian_two_sided_coverage(stddev_spacing: float) -> float:
    return float(math.erf(float(stddev_spacing) / math.sqrt(2.0)))


def gaussian_alpha_from_stddev_spacing(stddev_spacing: float) -> float:
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
    return ((x + period / 2.0) % period) - period / 2.0


def make_spd(cov: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    cov = cov.float()
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    p = cov.shape[-1]
    eye = torch.eye(p, dtype=cov.dtype, device=cov.device)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1, keepdim=True).clamp_min(eps)
    jitter = (eps + 1e-6 * trace / max(p, 1)).unsqueeze(-1) * eye
    return cov + jitter


def chi2_ppf_approx(prob: float, df: int) -> float:
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(prob, df))
    except Exception:
        try:
            from statistics import NormalDist
            z = NormalDist().inv_cdf(prob)
        except Exception:
            z = 0.475232
        return float(df * (1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))) ** 3)


def conformal_quantile(scores: torch.Tensor, alpha: float) -> torch.Tensor:
    s = scores.detach().flatten()
    s = s[torch.isfinite(s)]
    if s.numel() == 0:
        raise ValueError("No finite conformal scores were available.")
    n = s.numel()
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return torch.sort(s).values[k - 1]


# -----------------------------------------------------------------------------
# Source ordering / matching
# -----------------------------------------------------------------------------

def sorted_by_doa(
    mean_rad: torch.Tensor,
    cov_rad2: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    if mean_rad.shape != target_rad.shape:
        raise ValueError(
            f"mean and target shapes must match, got {mean_rad.shape} and {target_rad.shape}"
        )

    b, s, p = mean_rad.shape
    perms = torch.tensor(
        list(itertools.permutations(range(p))),
        device=mean_rad.device,
        dtype=torch.long,
    )
    nperm = perms.shape[0]

    mean_exp = mean_rad.unsqueeze(2).expand(b, s, nperm, p)
    perm_exp = perms.view(1, 1, nperm, p).expand(b, s, nperm, p)
    mean_perm = torch.gather(mean_exp, dim=3, index=perm_exp)

    err_perm = wrap_periodic_error(mean_perm - target_rad.unsqueeze(2), period=period)
    cost = torch.sum(err_perm ** 2, dim=-1)
    best_idx = torch.argmin(cost, dim=2)

    mean_best = torch.gather(
        mean_perm,
        dim=2,
        index=best_idx.view(b, s, 1, 1).expand(b, s, 1, p),
    ).squeeze(2)
    err_best = torch.gather(
        err_perm,
        dim=2,
        index=best_idx.view(b, s, 1, 1).expand(b, s, 1, p),
    ).squeeze(2)

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


# -----------------------------------------------------------------------------
# Model loading / covariance extraction
# -----------------------------------------------------------------------------

def extract_covariance_rad2(
    model_result: Dict[str, torch.Tensor],
    eps: float = 1e-10,
) -> torch.Tensor:
    if "covariance_i" in model_result and model_result["covariance_i"] is not None:
        return make_spd(model_result["covariance_i"].float(), eps=eps)

    if model_output_to_covariance_rad2 is not None:
        try:
            return make_spd(model_output_to_covariance_rad2(model_result).float(), eps=eps)
        except Exception:
            pass

    if "cov_i" in model_result and model_result["cov_i"] is not None:
        return make_spd(
            model_result["cov_i"].float() * ((math.pi / 180.0) ** 2),
            eps=eps,
        )

    if "sigma_i" not in model_result:
        raise KeyError("Model output does not contain covariance_i, cov_i, or sigma_i.")

    sigma_rad = torch.deg2rad(model_result["sigma_i"].float().clamp_min(eps))
    return make_spd(torch.diag_embed(sigma_rad ** 2), eps=eps)


def load_ssn_stage2_model(
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

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)

    # Give a clearer error when a standalone SignalsSubspaceNet checkpoint is
    # accidentally supplied instead of the full SSN Stage 2 checkpoint.
    if not any(k.startswith("subarray_models.") for k in state.keys()):
        raise ValueError(
            "The checkpoint does not look like a full MultiSubarraysModel/SSN Stage 2 "
            "checkpoint. Expected keys starting with 'subarray_models.'. Use "
            "best_base_reliability.pth from the multi_subarray run."
        )

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
    label: str = "Deterministic inference",
    progress_every_batches: int = 1,
) -> PredictionSet:
    """Collect deterministic predictions and emit regular batch-level progress."""
    model.eval()
    means: List[torch.Tensor] = []
    covs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    total_batches = len(loader)
    if total_batches == 0:
        raise ValueError(f"{label}: received an empty DataLoader.")

    progress(f"{label}: started ({total_batches} batch(es)).")
    for batch_idx, (sensor_positions, _source_positions, samples, doa_gt) in enumerate(loader, start=1):
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

        if (
            batch_idx == 1
            or batch_idx == total_batches
            or batch_idx % max(1, progress_every_batches) == 0
        ):
            progress(f"{label}: completed batch {batch_idx}/{total_batches}.")

    progress(f"{label}: complete.")
    return PredictionSet(
        mean_rad=torch.cat(means, dim=0),
        cov_rad2=torch.cat(covs, dim=0),
        target_rad=torch.cat(targets, dim=0),
    )


def enable_dropout_only(model: nn.Module) -> None:
    """Keep normalization layers in eval mode and activate dropout only."""
    model.eval()
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
        ) or "dropout" in class_name:
            module.train()


@torch.no_grad()
def mc_dropout_prediction_sets(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_mc: int,
    eps: float = 1e-10,
    label: str = "MC-Dropout inference",
    mc_progress_every: int = 25,
) -> MCDropoutPredictionSets:
    """
    Produce MC-Dropout statistics and emit MC-pass progress.

    Both comparison rows later use the diagonal covariance
    diag(Var_w[theta_hat_w]) from the same stochastic DOA outputs.
    """
    if n_mc < 2:
        raise ValueError("n_mc must be at least 2 to estimate MC-Dropout variance.")

    epi_means: List[torch.Tensor] = []
    epi_covs: List[torch.Tensor] = []
    alea_covs: List[torch.Tensor] = []
    total_covs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    total_batches = len(loader)
    if total_batches == 0:
        raise ValueError(f"{label}: received an empty DataLoader.")

    progress(
        f"{label}: started ({total_batches} batch(es), {n_mc} stochastic passes per batch, "
        f"heartbeat every {max(1, mc_progress_every)} passes)."
    )

    for batch_idx, (sensor_positions, _source_positions, samples, doa_gt) in enumerate(loader, start=1):
        sensor_positions = sensor_positions.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        doa_gt = torch.deg2rad(doa_gt.to(device, non_blocking=True)).float()

        pass_means: List[torch.Tensor] = []
        pass_covs: List[torch.Tensor] = []

        progress(f"{label}: batch {batch_idx}/{total_batches} started.")
        for mc_idx in range(1, n_mc + 1):
            enable_dropout_only(model)
            out = model(sensor_positions, samples, doa_gt)
            pred = out["bearings"].float()
            cov = extract_covariance_rad2(out, eps=eps)

            # Stabilize source identity across stochastic passes without using GT.
            pred, cov = sorted_by_doa(pred, cov)
            pass_means.append(pred)
            pass_covs.append(cov)

            if (
                mc_idx == 1
                or mc_idx == n_mc
                or mc_idx % max(1, mc_progress_every) == 0
            ):
                progress(
                    f"{label}: batch {batch_idx}/{total_batches}, "
                    f"MC pass {mc_idx}/{n_mc} complete."
                )

        model.eval()

        pred_stack = torch.stack(pass_means, dim=0)   # [T,B,S,P]
        cov_stack = torch.stack(pass_covs, dim=0)     # [T,B,S,P,P]

        mu = pred_stack.mean(dim=0)
        residual = wrap_periodic_error(
            pred_stack - mu.unsqueeze(0),
            period=math.pi,
        )
        cov_epistemic = torch.einsum(
            "tbsi,tbsj->bsij",
            residual,
            residual,
        ) / float(n_mc - 1)
        cov_epistemic = make_spd(cov_epistemic, eps=eps)

        cov_aleatoric = make_spd(cov_stack.mean(dim=0), eps=eps)
        cov_total = make_spd(cov_epistemic + cov_aleatoric, eps=eps)

        # Align once after all MC statistics have been formed.
        mu_aligned, epi_aligned, _ = align_to_target(
            mu,
            doa_gt,
            cov_epistemic,
        )
        _, alea_aligned, _ = align_to_target(
            mu,
            doa_gt,
            cov_aleatoric,
        )
        _, total_aligned, _ = align_to_target(
            mu,
            doa_gt,
            cov_total,
        )

        epi_means.append(mu_aligned.detach().cpu())
        epi_covs.append(epi_aligned.detach().cpu())
        alea_covs.append(alea_aligned.detach().cpu())
        total_covs.append(total_aligned.detach().cpu())
        targets.append(doa_gt.detach().cpu())
        progress(f"{label}: batch {batch_idx}/{total_batches} aggregation complete.")

    mean_all = torch.cat(epi_means, dim=0)
    target_all = torch.cat(targets, dim=0)

    progress(f"{label}: complete.")
    return MCDropoutPredictionSets(
        epistemic=PredictionSet(
            mean_rad=mean_all,
            cov_rad2=torch.cat(epi_covs, dim=0),
            target_rad=target_all,
        ),
        mean_aleatoric=PredictionSet(
            mean_rad=mean_all,
            cov_rad2=torch.cat(alea_covs, dim=0),
            target_rad=target_all,
        ),
        bayesian_total=PredictionSet(
            mean_rad=mean_all,
            cov_rad2=torch.cat(total_covs, dim=0),
            target_rad=target_all,
        ),
    )


# -----------------------------------------------------------------------------
# Absolute-residual source-wise split conformal prediction
# -----------------------------------------------------------------------------

def sourcewise_absolute_residual_scores(
    pred: PredictionSet,
    period: float = math.pi,
) -> torch.Tensor:
    """
    Source-wise nonconformity scores in radians:
        s_{i,k} = |wrap(theta_{i,k} - theta_hat_{i,k})|.

    Source identity is defined by ascending DoA order for both predictions and
    labels. This deterministic convention is available at inference time and
    avoids using a ground-truth-dependent Hungarian assignment to define the
    conformal source index. The function intentionally ignores pred.cov_rad2.
    """
    pred_sorted, _ = torch.sort(pred.mean_rad, dim=-1)
    target_sorted, _ = torch.sort(pred.target_rad, dim=-1)
    return torch.abs(wrap_periodic_error(pred_sorted - target_sorted, period=period))


def conformal_quantile_by_source(
    scores: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Finite-sample split-conformal order statistic, one q_alpha,k per source."""
    return torch.stack(
        [conformal_quantile(scores[..., k], alpha=alpha) for k in range(scores.shape[-1])],
        dim=0,
    )


def absolute_conformalize_sourcewise(
    cal_pred: PredictionSet,
    test_pred: PredictionSet,
    alpha: float,
    eps: float = 1e-10,
    period: float = math.pi,
) -> Tuple[PredictionSet, torch.Tensor]:
    """
    Build source-wise absolute-residual CP intervals:
        C_k(x) = [theta_hat_k(x)-q_alpha,k, theta_hat_k(x)+q_alpha,k].

    For covariance-oriented diagnostic metrics only, encode the interval
    half-widths as a constant diagonal matrix diag(q_alpha,k^2).  No Stage-1
    covariance is read, reused, or rescaled in this function.
    """
    scores = sourcewise_absolute_residual_scores(cal_pred, period=period)
    qhat_rad = conformal_quantile_by_source(scores, alpha=alpha).float()

    b, s, p = test_pred.mean_rad.shape
    if qhat_rad.numel() != p:
        raise ValueError(f"Expected {p} conformal half-widths, got {qhat_rad.numel()}.")

    # Use the same inference-available ascending-DoA source convention on the
    # final evaluation split. q_alpha,k is therefore associated with the k-th
    # ordered DoA, not with a ground-truth-dependent matching permutation.
    mean_sorted, _ = torch.sort(test_pred.mean_rad, dim=-1)
    target_sorted, _ = torch.sort(test_pred.target_rad, dim=-1)

    q = qhat_rad.to(test_pred.mean_rad.device).view(1, 1, p).expand(b, s, p)
    cov_cp = make_spd(torch.diag_embed(q.square()), eps=eps)

    return PredictionSet(
        mean_rad=mean_sorted,
        cov_rad2=cov_cp,
        target_rad=target_sorted,
    ), qhat_rad


# -----------------------------------------------------------------------------
# Metrics / output
# -----------------------------------------------------------------------------

def covariance_metrics(
    name: str,
    pred: PredictionSet,
    alpha: float,
    stddev_spacing: float,
    eps: float = 1e-10,
    period: float = math.pi,
) -> Dict[str, Any]:
    mean, cov, err = align_to_target(
        pred.mean_rad,
        pred.target_rad,
        pred.cov_rad2,
        period=period,
    )
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
    joint_coverage = (nees <= chi2_thr).float().mean()

    diag = torch.diagonal(flat_cov, dim1=-2, dim2=-1).clamp_min(eps)
    sigma = torch.sqrt(diag)
    sigma_deg = torch.rad2deg(sigma)

    source_hit = (
        torch.abs(flat_e) <= float(stddev_spacing) * sigma
    ).float()
    source_coverage = source_hit.mean()
    source_coverage_per_source = source_hit.reshape(n, s, p).mean(dim=(0, 1))

    apec = flat_cov.mean(dim=0)
    eec = (flat_e.T @ flat_e) / float(max(f, 1))
    apec_trace = torch.trace(apec)
    eec_trace = torch.trace(eec)
    trace_gap = torch.abs(apec_trace - eec_trace)
    fro_gap = torch.linalg.norm(apec - eec, ord="fro")
    eec_fro = torch.linalg.norm(eec, ord="fro").clamp_min(eps)
    fro_rel = fro_gap / eec_fro

    sign, logabsdet = torch.linalg.slogdet(flat_cov)
    nll_all = 0.5 * (nees + logabsdet + p * math.log(2.0 * math.pi))
    nll_finite = nll_all[torch.isfinite(nll_all)]
    nll = nll_finite.mean() if nll_finite.numel() else torch.tensor(float("nan"))
    valid_logdet = logabsdet[sign > 0]
    mean_logdet = (
        valid_logdet.mean()
        if valid_logdet.numel()
        else torch.tensor(float("nan"))
    )

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
        "coverage_chi2_target": float(joint_coverage.cpu()),
        "coverage_chi2_target_error": float(
            torch.abs(joint_coverage - target_coverage).cpu()
        ),
        "chi2_target_threshold": float(chi2_thr),
        "source_coverage_target": float(source_coverage.cpu()),
        "source_coverage_target_error": float(
            torch.abs(source_coverage - target_coverage).cpu()
        ),
        "source_coverage_per_source": _jsonify(source_coverage_per_source.cpu()),
        "mean_sigma_deg": float(sigma_deg.mean().cpu()),
        "median_sigma_deg": float(sigma_deg.median().cpu()),
        "mean_trace_cov_rad2": float(apec_trace.cpu()),
        "mean_trace_cov_deg2": float(
            (apec_trace * (180.0 / math.pi) ** 2).cpu()
        ),
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
        "anees_normalized",
        "mean_sigma_deg",
        "mean_trace_cov_deg2",
        "gaussian_nll",
        "apec_eec_rel",
        "qhat_deg",
    ]
    widths = {k: max(len(k), 12) for k in keys}
    for row in rows:
        for key in keys:
            value = row.get(key, "")
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            widths[key] = max(widths[key], len(text))

    print("\n" + " | ".join(k.ljust(widths[k]) for k in keys))
    print("-+-".join("-" * widths[k] for k in keys))
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            text = f"{value:.6g}" if isinstance(value, float) else str(value)
            cells.append(text.ljust(widths[key]))
        print(" | ".join(cells))


def write_outputs(
    out_dir: str,
    rows: List[Dict[str, Any]],
    tensors: Dict[str, PredictionSet],
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ssn_stage1_absolute_cp_vs_stage2_method_comparison.csv")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(out_dir, "ssn_stage1_absolute_cp_vs_stage2_method_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(rows), f, indent=2)

    pt_path = os.path.join(out_dir, "ssn_stage1_absolute_cp_vs_stage2_method_tensors.pt")
    torch.save(
        {
            key: {
                "mean_rad": value.mean_rad,
                "cov_rad2": value.cov_rad2,
                "target_rad": value.target_rad,
            }
            for key, value in tensors.items()
        },
        pt_path,
    )

    print(f"\nSaved:\n  {csv_path}\n  {json_path}\n  {pt_path}")



# -----------------------------------------------------------------------------
# Stage 1 baselines vs Stage 2 proposed comparison
# -----------------------------------------------------------------------------

def load_ssn_model_from_checkpoint(
    args: argparse.Namespace,
    dataset: SensorSourceGraphDataset,
    checkpoint_path: str,
    device: torch.device,
    checkpoint_label: str,
) -> nn.Module:
    """Load a full MultiSubarraysModel checkpoint for either Stage 1 or Stage 2."""
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)

    if not any(k.startswith("subarray_models.") for k in state.keys()):
        raise ValueError(
            f"{checkpoint_label} does not look like a full MultiSubarraysModel checkpoint. "
            "Expected keys starting with 'subarray_models.'."
        )

    model.load_state_dict(state)
    model.eval()
    return model


def diagonalize_prediction_covariance(pred: PredictionSet, eps: float = 1e-10) -> PredictionSet:
    """Keep only the marginal MC-Dropout variances as a diagonal covariance."""
    diag = torch.diagonal(pred.cov_rad2, dim1=-2, dim2=-1).clamp_min(eps)
    return PredictionSet(
        mean_rad=pred.mean_rad,
        cov_rad2=make_spd(torch.diag_embed(diag), eps=eps),
        target_rad=pred.target_rad,
    )


def main(argv: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    parser = argparse.ArgumentParser(
        "Compare Stage-1 MC/Bayesian and absolute-residual CP baselines against Stage-2 covariance"
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument(
        "--stage1_checkpoint_path",
        required=True,
        help="Stage-1/base-accuracy SubspaceNet checkpoint, normally best_base_accuracy.pth",
    )
    parser.add_argument(
        "--stage2_checkpoint_path",
        required=True,
        help="Stage-2/reliability SubspaceNet checkpoint, normally best_base_reliability.pth",
    )
    parser.add_argument("--test_dataset_path", required=True)
    parser.add_argument(
        "--cp_calibration_fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of the independently generated test_dataset.pt reserved exclusively "
            "for absolute-residual CP calibration. The remaining samples are the final "
            "evaluation set for every compared method."
        ),
    )
    parser.add_argument("--out_dir", default="uq_ssn_stage1_baselines_vs_stage2")
    parser.add_argument("--stage1_name", default="SubspaceNet Stage 1")
    parser.add_argument("--stage2_name", default="SubspaceNet Stage 2")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_mc", type=int, default=200)
    parser.add_argument("--stddev_spacing", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument(
        "--progress_every_batches",
        type=int,
        default=1,
        help="Emit deterministic-inference progress every N batches.",
    )
    parser.add_argument(
        "--mc_progress_every",
        type=int,
        default=25,
        help="Emit MC-Dropout progress every N stochastic passes.",
    )
    parser.add_argument(
        "--include_stage1_regular",
        action="store_true",
        help="Also write the deterministic Stage-1 covariance row for debugging.",
    )

    # Attributes required by MultiSubarraysModel.
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
    progress(f"Stage 1 checkpoint: {args.stage1_checkpoint_path}")
    progress(f"Stage 2 checkpoint: {args.stage2_checkpoint_path}")
    progress(f"MC passes: {args.n_mc}")
    progress(
        f"Gaussian spacing: +/-{args.stddev_spacing:g} sigma, "
        f"target coverage={1.0 - args.alpha:.9f}, alpha={args.alpha:.9f}"
    )

    test_ds: SensorSourceGraphDataset = torch.load(
        args.test_dataset_path,
        weights_only=False,
    )
    test_ds.set_use_graph_features(use_features=True)

    if not (0.0 < args.cp_calibration_fraction < 1.0):
        raise ValueError("cp_calibration_fraction must be strictly between 0 and 1.")

    # IMPORTANT: test_dataset.pt must itself be independent of training/checkpoint
    # selection. We randomly reserve 20% of it only for CP calibration and never
    # report any final metric on those calibration scenes.
    n_total = len(test_ds)
    n_cal = int(round(n_total * args.cp_calibration_fraction))
    n_cal = max(1, n_cal)
    if n_cal >= n_total:
        raise ValueError("CP calibration split leaves no final evaluation samples.")

    split_gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=split_gen)
    cal_indices = perm[:n_cal].tolist()
    eval_indices = perm[n_cal:].tolist()
    cal_ds = Subset(test_ds, cal_indices)
    eval_ds = Subset(test_ds, eval_indices)

    progress(
        f"CP split of independent test dataset: {n_cal}/{n_total} "
        f"({100.0 * n_cal / n_total:.1f}%) calibration scenes; "
        f"{len(eval_indices)}/{n_total} final-evaluation scenes. "
        f"Random seed={args.seed}."
    )

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

    progress("Loading Stage-1/base-accuracy SubspaceNet model...")
    stage1_model = load_ssn_model_from_checkpoint(
        args,
        test_ds,
        args.stage1_checkpoint_path,
        device,
        checkpoint_label=args.stage1_name,
    )

    progress("Loading Stage-2/proposed-reliability SubspaceNet model...")
    stage2_model = load_ssn_model_from_checkpoint(
        args,
        test_ds,
        args.stage2_checkpoint_path,
        device,
        checkpoint_label=args.stage2_name,
    )

    progress("[1/5] Stage-1 deterministic DoA on CP calibration data...")
    stage1_cal = deterministic_predictions(
        stage1_model, cal_loader, device, eps=args.eps,
        label="[1/5] Stage-1 calibration inference",
        progress_every_batches=args.progress_every_batches,
    )

    progress("[2/5] Stage-1 deterministic DoA on final evaluation data...")
    stage1_test = deterministic_predictions(
        stage1_model, test_loader, device, eps=args.eps,
        label="[2/5] Stage-1 test inference",
        progress_every_batches=args.progress_every_batches,
    )

    progress(f"[3/5] Stage-1 MC-Dropout on final evaluation data, n_mc={args.n_mc}...")
    stage1_mc_sets = mc_dropout_prediction_sets(
        stage1_model,
        test_loader,
        device,
        n_mc=args.n_mc,
        eps=args.eps,
        label="[3/5] Stage-1 MC-Dropout",
        mc_progress_every=args.mc_progress_every,
    )

    # Both Stage-1 MC-based baselines deliberately use diagonal covariance only:
    # diag(Var_w[theta_hat_w]). This matches a per-source MC-Dropout/Bayesian
    # uncertainty baseline and avoids evaluating cross-source correlations for
    # either baseline.
    stage1_mc_diag = diagonalize_prediction_covariance(stage1_mc_sets.epistemic, eps=args.eps)
    stage1_bayesian_diag = diagonalize_prediction_covariance(stage1_mc_sets.epistemic, eps=args.eps)

    progress("[4/5] Absolute-residual source-wise CP: q from 20% calibration split; no Stage-1 covariance used...")
    stage1_cp_test, qhat_rad = absolute_conformalize_sourcewise(
        stage1_cal,
        stage1_test,
        alpha=args.alpha,
        eps=args.eps,
    )

    progress("[5/5] Stage-2 proposed full covariance on final evaluation data...")
    stage2_test = deterministic_predictions(
        stage2_model, test_loader, device, eps=args.eps,
        label="[5/5] Stage-2 test inference",
        progress_every_batches=args.progress_every_batches,
    )

    stage1_regular_name = f"{args.stage1_name} (regular)"
    mc_name = f"{args.stage1_name} + MC Dropout"
    bayes_name = f"{args.stage1_name} + Bayesian MC Dropout"
    cp_name = f"{args.stage1_name} + conformal prediction"
    proposed_name = f"{args.stage2_name} (proposed)"

    rows: List[Dict[str, Any]] = []
    tensors: Dict[str, PredictionSet] = {
        mc_name: stage1_mc_diag,
        bayes_name: stage1_bayesian_diag,
        cp_name: stage1_cp_test,
        proposed_name: stage2_test,
    }

    if args.include_stage1_regular:
        stage1_regular_row = covariance_metrics(
            stage1_regular_name,
            stage1_test,
            alpha=args.alpha,
            stddev_spacing=args.stddev_spacing,
            eps=args.eps,
        )
        stage1_regular_row["covariance_definition"] = "Stage-1 deterministic model covariance"
        stage1_regular_row["checkpoint_role"] = "baseline_stage1"
        rows.append(stage1_regular_row)
        tensors[stage1_regular_name] = stage1_test

    mc_row = covariance_metrics(
        mc_name,
        stage1_mc_diag,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    )
    mc_row["covariance_definition"] = "diag(Var_w[theta_hat_w]) from Stage-1 MC Dropout"
    mc_row["checkpoint_role"] = "baseline_stage1"
    mc_row["n_mc"] = args.n_mc
    mc_row["diagonal_covariance_only"] = True
    rows.append(mc_row)

    bayes_row = covariance_metrics(
        bayes_name,
        stage1_bayesian_diag,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    )
    bayes_row["covariance_definition"] = "diag(Var_w[theta_hat_w]) from Stage-1 stochastic DOA outputs"
    bayes_row["checkpoint_role"] = "baseline_stage1"
    bayes_row["n_mc"] = args.n_mc
    bayes_row["diagonal_covariance_only"] = True
    rows.append(bayes_row)

    cp_row = covariance_metrics(
        cp_name,
        stage1_cp_test,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    )
    cp_row["covariance_definition"] = (
        "diag(q_alpha,k^2): absolute-residual source-wise split conformal; "
        "Stage-1 covariance is not used"
    )
    cp_row["checkpoint_role"] = "baseline_stage1"
    cp_row["conformal_mode"] = "per_source_absolute_residual"
    cp_row["conformal_base"] = "Stage-1 deterministic DoA only; no Stage-1 covariance"
    cp_row["qhat_rad"] = _jsonify(qhat_rad.cpu())
    cp_row["qhat_deg"] = _jsonify(torch.rad2deg(qhat_rad).cpu())
    cp_row["cp_interval_definition"] = "theta_hat_(ordered,k) +/- q_alpha,k"
    cp_row["cp_source_association"] = "ascending DoA order; no ground-truth matching used to define CP source index"
    cp_row["cp_calibration_fraction"] = float(args.cp_calibration_fraction)
    cp_row["cp_calibration_num_scenes"] = int(n_cal)
    cp_row["cp_evaluation_num_scenes"] = int(len(eval_indices))
    cp_row["cp_calibration_source"] = "random 20% split of independent test_dataset.pt"
    cp_row["joint_metrics_note"] = (
        "Diagnostic only: source-wise absolute CP guarantees marginal coverage, "
        "not chi-square joint-region coverage."
    )
    rows.append(cp_row)

    proposed_row = covariance_metrics(
        proposed_name,
        stage2_test,
        alpha=args.alpha,
        stddev_spacing=args.stddev_spacing,
        eps=args.eps,
    )
    proposed_row["covariance_definition"] = "Stage-2 deterministic proposed covariance"
    proposed_row["checkpoint_role"] = "proposed_stage2"
    rows.append(proposed_row)

    progress(
        "Note: Stage-1 MC-Dropout and Bayesian MC-Dropout both use "
        "diag(Var_w[theta_hat_w]) from the same stochastic passes; under the "
        "requested diagonal-only definition they are expected to be identical."
    )
    progress(
        "CP is absolute-residual source-wise split conformal: q_alpha,k is an "
        "empirical half-width in radians, and the CP row does not consume or scale "
        "the Stage-1 covariance."
    )
    for row in rows:
        row["cp_calibration_fraction"] = float(args.cp_calibration_fraction)
        row["cp_calibration_num_scenes"] = int(n_cal)
        row["final_evaluation_num_scenes"] = int(len(eval_indices))
        row["total_independent_test_scenes"] = int(n_total)
        row["test_split_seed"] = int(args.seed)
        row["reported_on_final_evaluation_split_only"] = True

    print_table(rows)
    progress("Metric table computed; writing files.")

    os.makedirs(args.out_dir, exist_ok=True)
    split_metadata = {
        "test_dataset_path": os.path.abspath(args.test_dataset_path),
        "cp_calibration_fraction_requested": float(args.cp_calibration_fraction),
        "cp_calibration_num_scenes": int(n_cal),
        "final_evaluation_num_scenes": int(len(eval_indices)),
        "total_independent_test_scenes": int(n_total),
        "split_seed": int(args.seed),
        "calibration_indices": cal_indices,
        "evaluation_indices": eval_indices,
        "note": (
            "Calibration indices are used only to calculate absolute-residual CP q_alpha,k. "
            "All reported method metrics are evaluated exclusively on evaluation_indices. "
            "CP source identity is the ascending DoA order."
        ),
    }
    with open(os.path.join(args.out_dir, "cp_test_split_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonify(split_metadata), f, indent=2)
    torch.save(
        {"calibration_indices": cal_indices, "evaluation_indices": eval_indices},
        os.path.join(args.out_dir, "cp_test_split_indices.pt"),
    )
    # Reuse write_outputs, then copy to a clearer filename.
    write_outputs(args.out_dir, rows, tensors)

    # Also write aliases with the comparison-specific names.
    csv_path = os.path.join(args.out_dir, "ssn_stage1_absolute_cp_vs_stage2_comparison.csv")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.out_dir, "ssn_stage1_absolute_cp_vs_stage2_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(rows), f, indent=2)

    progress(f"Comparison files written: {csv_path} | {json_path}")
    return rows


if __name__ == "__main__":
    main()
