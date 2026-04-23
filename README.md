# LLM Time-Series Adversarial Audit

This repository is a clean codebase for the `Code / Tool Submission` part of the project
proposal, **Evaluating and Mitigating Black-Box Adversarial Vulnerabilities in LLM-Based
Time Series Forecasting**.

The repository focuses on three things:

1. Reproducing a **black-box targeted attack** in the spirit of Directional Gradient
   Approximation (DGA).
2. Comparing **autoregressive** and **joint forecasting** architectures under the same
   attack pipeline.
3. Packaging the work as a **usable auditing tool** rather than a one-off notebook.

## What This Repo Includes

- Config-driven experiment runner.
- Time-series dataset loading, chronological splits, scaling, and sliding-window generation.
- A query-based targeted attack module (`DGAStyleAttack`) with `l1`, `l2`, and `linf`
  perturbation constraints.
- A lightweight `Ridge` baseline for sanity checks.
- Optional PyTorch baselines for:
  - `AutoregressiveTransformerForecaster`
  - `JointTransformerForecaster`
- Metric and reporting utilities for clean-vs-attacked evaluation.
- Tests for the core attack and metric logic.
- Documentation that explicitly states what the group implemented.

## Repository Layout

```text
llm_ts_adversarial_audit/
├── configs/
├── docs/
├── outputs/
├── scripts/
├── src/llm_ts_audit/
│   ├── attacks/
│   ├── data/
│   ├── evaluation/
│   └── models/
└── tests/
```

## Setup

### Minimal setup

This is enough for the synthetic debug pipeline and the included tests:

```bash
pip install -e .
```

### With PyTorch models

If you want to run the transformer baselines:

```bash
pip install -e ".[torch]"
```

### Developer/test setup

```bash
pip install -e ".[dev]"
```

## Quick Start

### 0. Download the benchmark datasets

The proposal datasets can be downloaded directly into `./data` with:

```bash
python scripts/download_benchmarks.py
```

This fetches:

- `data/ETTh1.csv`
- `data/weather.csv`
- `data/exchange_rate.csv`

### 1. Run a fully local synthetic sanity check

This works without downloading external datasets and is the fastest way to verify the
tool chain:

```bash
python scripts/run_experiment.py --config configs/synthetic_linear.yaml
```

### 2. Run a transformer comparison

```bash
python scripts/run_experiment.py --config configs/synthetic_autoregressive_transformer.yaml
python scripts/run_experiment.py --config configs/synthetic_joint_transformer.yaml
```

### 3. Run on a real benchmark CSV

First download the benchmarks:

```bash
python scripts/download_benchmarks.py
```

Then run:

```bash
python scripts/run_experiment.py --config configs/etth1_autoregressive_transformer.yaml
```

## Expected Dataset Format

The loader expects a CSV file with a timestamp column optionally present and one or more
numeric feature columns. If `feature_columns` is not supplied, the loader will select all
numeric columns except the declared time column. If the configured time column is missing,
the loader will continue by treating the file as a purely numeric multivariate series.

This makes the repo usable for:

- ETTh1
- Weather
- Exchange
- other benchmark CSVs with similar tabular layouts

## Output Files

Each run creates a dedicated directory under `outputs/` containing:

- `summary.json`: aggregate metrics and experiment metadata
- `summary.md`: readable experiment summary
- `per_sample_metrics.csv`: sample-level clean vs attacked results
- `resolved_config.yaml`: the exact config used

If attacks are enabled, runs now also create `outputs/<run>/interpretability/` with:

- `horizon_error_profile.csv`: error increase at each forecast step
- `input_block_sensitivity.csv`: per-sample block masking effects
- `input_block_sensitivity_summary.csv`: average block sensitivity
- `representative_forecasts.csv`: representative clean vs attacked traces
- `summary.md`: interpretability-oriented summary text

## Attack Design

The implemented attack is a **DGA-style targeted black-box attack**:

- It never uses model gradients directly.
- It estimates a useful update direction from query responses to random probing signals.
- It targets a **noise-like sequence** by default, matching the proposal's focus on
  degraded, anomalous forecasts.
- It supports configurable query budgets, target generation, perturbation budgets, and norms.

## What The Group Implemented

See [IMPLEMENTED_BY_GROUP.md](docs/IMPLEMENTED_BY_GROUP.md). It contains the scope of this project and responsibilities of each member

## Suggested Submission Checklist

- README is present with setup and usage instructions.
- Code is modular and not notebook-dependent.
- Implemented components are clearly documented.
- Experiments are reproducible from YAML configs.
- The attack pipeline and model pipeline can both be demonstrated.

## Current Limitations

- The transformer baselines are lightweight research baselines, not production
  re-implementations of TimeGPT or LLMTime.
- No external API model is bundled, because reproducibility and credentials vary.
- Attack success on real benchmarks still depends on dataset choice, tuning, and available
  compute.

## Dataset Sources

The downloader script uses a public Hugging Face mirror of the commonly used Autoformer /
TSLib benchmark files for convenience:

- [ETTh1.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/blob/main/ETTh1.csv)
- [weather.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/blob/main/weather.csv)
- [exchange_rate.csv](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets/blob/main/exchange_rate.csv)

The mirrored dataset card also includes `wget` examples for these benchmark files:
[dataset card](https://huggingface.co/datasets/pkr7098/time-series-forecasting-datasets).
