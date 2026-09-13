# Pretrained source-sweep comparison

The runner evaluates **SubspaceNet, TransMUSIC, DataDrivenComplex, and classical ESPRIT** on one shared set of native simulated observations per condition. Each learned checkpoint is loaded once and evaluated with frozen parameters. It exports combined comparison CSVs, per-model CSVs, per-trial CSVs, raw NPZ predictions and publication figures. It does not train or overwrite checkpoints.

## Install using your server layout

The ZIP is arranged to match your screenshot. Extract it into **multi_rssn/**:

- `MBDL_MultiSubArrays/run_source_sweep_comparison.py`
- `MBDL_MultiSubArrays/inter_source_tools/`
- `MBDL_MultiSubArrays/source_sweep_models.example.json`
- `sweep_non_coherent/source_sweep_comparison.sbatch`

The Python file belongs at the project root, alongside `trainer.py` and `src/`. The job file belongs in the sibling `sweep_non_coherent/` folder. The package includes all helper modules, so the older sweep package is not required. Your existing native `src/` is required and is not replaced.

From **multi_rssn/**:

```bash
mkdir -p job_logs
sbatch sweep_non_coherent/source_sweep_comparison.sbatch
```

The launcher uses your `main` partition, `rtx_6000`, 50 GB memory, eight CPUs and `remote_subspace` environment. Submit from `multi_rssn/`; alternatively set `PROJECT_DIR` to the absolute path of `MBDL_MultiSubArrays`. If using `rtx_4090`, update both the GPU and constraint directives.

## Existing checkpoint selection

Default paths follow your source-count sweeps:

```
sweeps/<model>/sweep_subarray_config_0_system_model_M/
    subarray_config_0_system_model_M_2/<model>/best_base_reliability.pth
```

Each run's `config.json` must be in the value directory above `<model>/`. The three model names are `multi_subarray`, `transmusic`, and `data_driven_complex`. These files must already exist on your server; the supplied archives contained no pretrained weights.

If any checkpoint or original config is stored elsewhere, edit `source_sweep_models.example.json` with the exact paths. Paths are interpreted relative to the project root. This supports the separate checkpoint/config directories shown in your existing SNR benchmark jobs. Then submit from `multi_rssn/`:

```bash
export MODELS_JSON="source_sweep_models.example.json"
sbatch sweep_non_coherent/source_sweep_comparison.sbatch
```

You can select different stage filenames per model in that JSON. To use the same other filename for all default-layout runs, set `CHECKPOINT_NAME`, for example `base_reliability_completed.pth`.

All configurations must describe **L=1, M=2**, with the same N and T. The runner also checks source statistics, SNR and mismatch settings. It stops on a mismatch rather than silently using different test observations. If a deliberate common out-of-distribution evaluation is intended, provide `--simulation_config matching_evaluation_config.json`; N/M/T must still match all checkpoints, and differences from each training configuration are recorded.

For a preflight without loading PyTorch, from `MBDL_MultiSubArrays/`:

```bash
python run_source_sweep_comparison.py \
  --models_json source_sweep_models.example.json \
  --dense_near_anchor --dry_run
```

Preflight verifies paths/configurations; actual state-dict compatibility is checked during real inference. A small real-checkpoint smoke run is useful before the full sweep:

```bash
python run_source_sweep_comparison.py \
  --moving_values 35 45 --trials 32 --batch_size 16 \
  --results_dir sweep_results/inter_source_smoke
```

## Experiments

The job launcher runs all five experiments, including the four-method comparison and each method's full/diagonal/sign-reversed covariance ablations:

| Experiment | Evaluation conditions |
|---|---|
| `fixed_anchor` | Source 1 fixed at 40 degrees, source 2 from -90 to 90 in 5-degree steps, with extra points near 40 |
| `symmetric_separation` | Pair centered at 0 degrees, gaps 1,2,3,5,8,12,20,30 degrees |
| `fixed_separation` | Gap 10 degrees, centers -80 to 80 in 20-degree steps |
| `coherence` | Pair 40/35 degrees; Gaussian waveform correlation 0,.5,.9,.99,1 |
| `power_imbalance` | Pair 40/35 degrees; moving-source attenuation 0,5,10,15,20 dB |

There are 72 conditions per SNR for the full dense suite. With 5,000 trials and four methods, that is 1,440,000 trial predictions. The Python CLI defaults to `fixed_anchor`; the job file explicitly selects all five experiments.

For only the main sweep, from `multi_rssn/`:

```bash
sbatch sweep_non_coherent/source_sweep_comparison.sbatch \
  --experiments fixed_anchor
```

Or run directly from `MBDL_MultiSubArrays/`:

```bash
python run_source_sweep_comparison.py \
  --model_types multi_subarray transmusic data_driven_complex \
  --include_esprit --experiments fixed_anchor \
  --fixed_source_deg 40 --moving_start -90 --moving_stop 90 --moving_step 5 \
  --dense_near_anchor --trials 5000 --batch_size 256 \
  --results_dir sweep_results/inter_source_comparison
```

Use `--snr_values ...` for evaluation-SNR conditions, `--no-include_esprit` to omit the analytic baseline, or `--model_types ...` for a subset. An explicit `--models_json` determines the learned model list instead of `--model_types`. Use a fresh results directory when changing the experiment definition.

## Exported results

Everything is placed under **sweep_results/inter_source_comparison/** by default, controlled by `--results_dir`, as in your run-sweep workflow.

| Output | Contents |
|---|---|
| `benchmark_results.csv` | Combined long-form summary: one row per condition and model |
| `models/<label>/benchmark_results.csv` | The same summary, split by checkpoint/model |
| `trial_results/<case_id>/<label>.csv` | One row per Monte Carlo trial: true/estimated angles, aligned errors, covariance entries, validity flags and each variant's NEES |
| `numerical_results/all_models_<experiment>_snr_<value>.csv` | Wide-form plot table: model curves aligned by condition |
| `cases/<case_id>/<label>.npz` | Raw predictions and covariance plus aligned errors, covariance and NEES arrays |
| `figures/all_models_*.png` and `*.pdf` | Overlaid method-comparison figures and per-model correlation comparisons |
| `figures/<model-specific-name>.*` | Full/diagonal/sign-flipped ablation figures and selected covariance ellipses |
| `plotting_latex_code/*.tex` | Matching editable PGFPlots figure code |
| `plotting_latex_code/all_benchmark_figures_pgfplots.tex` | Master input for combined method-comparison figures |
| `sweep_plan.json`, `manifest.json` | Configuration, checkpoint paths/hashes, code hashes and numerical settings |

Summary CSV fields include `RMSE_anchor_deg`, `RMSE_moving_deg`, biases, predicted/empirical correlation, empirical and predicted covariance entries in deg², normalized ANEES, joint coverage, NLL, gap/midpoint calibration, invalid covariance fraction and paired NLL gains of full covariance over its ablations.

The trial CSV exports covariance in **deg²** and angles/errors in **degrees**. NEES is dimensionless and is the unnormalized two-source quadratic form; the summary normalized ANEES is its mean divided by two. Raw covariance entries preserve asymmetry if present. Used covariance is symmetrized, with any explicit projection separately documented. JSON and CSV can contain NaN for undefined diagnostics.

Trial CSV export is enabled by default and can be large. Use `--no-trial_csv` to retain only summary CSVs plus raw NPZ data. Files are saved condition by condition. Repeat an identical command with `--resume` to continue completed cases; the checkpoint/code/configuration signature must match. All selected models are held on the inference device; reduce the selected model list if their weights exceed device memory, and batch/chunk sizes for inference memory. Separate runs require matching settings and seeds for the same observations.

## Comparison interpretation

The main combined figure overlays all methods for fixed/first-source RMSE, moving/second-source RMSE, joint coverage, normalized ANEES, gap standard-deviation calibration, and invalid covariance fraction. Another combined figure compares each method's predicted error correlation to **that method's own empirical error correlation**.

The native TransMUSIC and DataDrivenComplex uncertainty heads are diagonal, so their three covariance ablations coincide. ESPRIT uses the native `esprit_batched` and full analytic uncertainty evaluated at **estimated** angles, with centered sample covariance. Your model uses its native full covariance extraction. Point-estimation errors stay unchanged when a model's covariance is ablated.

Covariance metrics use each model's valid covariance subset. A model with many invalid outputs must not be judged by conditional coverage alone. `joint95_operational_all_*` counts invalid outputs as uncovered and is also exported; this combines availability and coverage. Default policy reports invalidity without silently repairing it. Explicit `--covariance_policy project` is a separately labeled repair sensitivity experiment.

Preserved native details: modulo-pi error matching (unwrapped errors also summarized), amplitude `10**(snr/10)` with realized source/noise ratios recorded, and batch-wide learned centering. Keep batch size fixed across comparisons. Coincident angles and endfire are tagged and omitted from regular Gaussian UQ curves. Close-source matching can induce dependence, and the graph training generator's hardcoded 0.2-radian minimum makes small gaps a training-support stress test. Full covariance is extracted analytically; the supplied native training uncertainty loss uses diagonal terms and the analytic block runs without gradients.

## LaTeX and validation

Copy `numerical_results/` and `plotting_latex_code/` into the paper root. In the preamble:

```latex
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
```

In the document:

```latex
\input{plotting_latex_code/all_benchmark_figures_pgfplots.tex}
```

Alternatively, compile `all_benchmark_figures_standalone.tex` from the result directory. To regenerate figures without inference:

```bash
python run_source_sweep_comparison.py --plots_only \
  --results_dir sweep_results/inter_source_comparison
```

Validation covers numerical covariance diagnostics, native NumPy simulation, checkpoint/config preflight, identical inputs across methods, one predictor load per target, combined/per-trial CSV export, resume behavior, and unchanged checkpoint files. The orchestration test uses explicitly fake predictors; it is not neural-model validation. Generated comparison LaTeX is checked with synthetic software fixtures. No benchmark results or fake pretrained weights are included. Actual neural inference still requires the server's PyTorch environment and your trained checkpoints.
