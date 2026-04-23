# Implemented By The Group

This file is included to satisfy the assignment requirement that the repository must clearly indicate what the group implemented.

## Implemented In This Repository

- A config-driven robustness auditing pipeline for time-series forecasting.
- A DGA-style black-box attack module for query-based adversarial evaluation.
- Support for `l1`, `l2`, and `linf` perturbation constraints.
- Chronological data splitting, normalization, and sliding-window preparation for forecasting tasks.
- Evaluation utilities for clean MSE, clean MAE, attacked MSE, attacked MAE, perturbation statistics, and a whiteness score.
- Baseline forecasting models: `LinearForecaster`, `AutoregressiveTransformerForecaster`, and `JointTransformerForecaster`.
- YAML-based experiment configuration and output artifact generation.
- Interpretability outputs including horizon-wise error profiles, input-block sensitivity summaries, and representative forecast traces.
- Tests for core metric and attack behavior.

## Not Claimed As Original Implementation

- Public benchmark datasets such as ETTh1, Weather, and Exchange.
- The original DGA paper or its exact codebase.
- Commercial or closed model APIs such as TimeGPT.

## Scope Statement

- This repository is best understood as a research tool submission.
- It is aligned with the proposal's threat model.
- It is designed for reproducible experiments and class presentation.
- It is not presented as a full reproduction of every model in the baseline literature.

## Suggested Group Contribution Breakdown

Replace the placeholders below with the actual group member names before submission.

- `Person 1`: Led project coordination, attack-pipeline integration, experiment orchestration, and final synthesis of results across datasets.
- `Person 2`: Managed dataset preparation, benchmark setup, YAML configuration updates, and execution of the main training and attack runs.
- `Person 3`: Worked on evaluation logic, metric analysis, interpretability outputs, and comparative analysis of autoregressive versus joint robustness.
- `Person 4`: Focused on report writing, related-work and methodology documentation, presentation preparation, and final project packaging.

## Collaboration Note

- All four group members discussed the research question together.
- All four reviewed experiment outputs jointly.
- All four contributed to the interpretation of results.
- All four participated in the final presentation and submission.
