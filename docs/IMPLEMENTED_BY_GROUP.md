# Implemented By The Group

This file is included to satisfy the assignment requirement that the repository must
clearly indicate what the group implemented.

## Implemented in this repository

- A config-driven **robustness auditing pipeline** for time-series forecasting.
- A **DGA-style black-box targeted attack** module that:
  - queries the model only through predictions
  - estimates update directions from random perturbation probes
  - supports `l1`, `l2`, and `linf` perturbation constraints
  - measures per-sample query counts and perturbation norms
- Chronological data splitting, normalization, and sliding-window preparation for
  forecasting tasks.
- Evaluation utilities for:
  - clean MSE / MAE
  - attacked MSE / MAE
  - perturbation size statistics
  - a simple whiteness score to quantify noise-like collapse
- Baseline forecasting models:
  - `LinearForecaster`
  - `AutoregressiveTransformerForecaster`
  - `JointTransformerForecaster`
- YAML-based experiment configuration and output artifact generation.
- Tests for core metric and attack behavior.

## Not claimed as original implementation

- Public benchmark datasets such as ETTh1, Weather, and Exchange.
- The original DGA paper or its exact codebase.
- Commercial or closed model APIs such as TimeGPT.

## Scope statement

This repository is best understood as:

- a **research tool submission**
- aligned with the proposal's threat model
- designed for reproducible experiments and class presentation

It is not presented as a full reproduction of every model in the baseline literature.

