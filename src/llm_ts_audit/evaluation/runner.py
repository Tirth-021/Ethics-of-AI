from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from llm_ts_audit.attacks.dga import DGAStyleAttack
from llm_ts_audit.attacks.objectives import objective_direction
from llm_ts_audit.config import FullConfig, save_config
from llm_ts_audit.data.loading import PreparedDataset, load_prepared_dataset
from llm_ts_audit.evaluation.metrics import summarize_regression
from llm_ts_audit.interpretability.analysis import (
    AttackInterpretabilityBundle,
    generate_interpretability_outputs,
)
from llm_ts_audit.models.factory import build_model


def run_experiment(config: FullConfig) -> Path:
    _set_seed(config.experiment.seed)
    dataset = load_prepared_dataset(config.dataset, seed=config.experiment.seed)
    model = build_model(
        config=config.model,
        context_length=config.dataset.context_length,
        horizon=config.dataset.horizon,
        n_features=len(dataset.feature_columns),
    )

    model.fit(
        dataset.train.inputs,
        dataset.train.targets,
        dataset.val.inputs,
        dataset.val.targets,
    )

    output_dir = _prepare_output_dir(config)
    save_config(config, output_dir / "resolved_config.yaml")
    summary = _evaluate_and_save(config, dataset, model, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(_render_summary_markdown(summary), encoding="utf-8")
    return output_dir


def _prepare_output_dir(config: FullConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.experiment.output_dir) / f"{config.experiment.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _evaluate_and_save(
    config: FullConfig,
    dataset: PreparedDataset,
    model,
    output_dir: Path,
) -> dict[str, object]:
    test_inputs = dataset.test.inputs
    test_targets = dataset.test.targets
    clean_predictions = model.predict(test_inputs)
    clean_metrics_all = summarize_regression(test_targets, clean_predictions)

    summary: dict[str, object] = {
        "experiment": asdict(config.experiment),
        "dataset": {
            **asdict(config.dataset),
            "feature_columns": dataset.feature_columns,
            "metadata": dataset.metadata,
        },
        "model": {
            "kind": config.model.kind,
            "params": config.model.params,
            "resolved_name": model.name,
        },
        "clean_test_metrics": clean_metrics_all,
    }

    if not config.attack.enabled:
        return summary

    attack = DGAStyleAttack(config.attack)
    attack_count = min(config.attack.max_samples, len(test_inputs))
    objective_goal = objective_direction(config.attack.objective)
    attacked_records: list[dict[str, float | int]] = []
    attacked_clean_inputs: list[np.ndarray] = []
    attacked_adv_inputs: list[np.ndarray] = []
    attacked_targets: list[np.ndarray] = []
    attacked_clean_predictions: list[np.ndarray] = []
    attacked_adv_predictions: list[np.ndarray] = []

    for idx in range(attack_count):
        result = attack.run(
            model=model,
            clean_input=test_inputs[idx],
            clean_prediction=clean_predictions[idx],
            ground_truth=test_targets[idx],
            sample_seed=config.attack.seed + idx,
        )
        y_true = test_targets[idx]
        clean_metrics = summarize_regression(y_true, result.clean_prediction)
        adv_metrics = summarize_regression(y_true, result.adversarial_prediction)
        attacked_clean_inputs.append(test_inputs[idx])
        attacked_adv_inputs.append(result.adversarial_input)
        attacked_targets.append(y_true)
        attacked_clean_predictions.append(result.clean_prediction)
        attacked_adv_predictions.append(result.adversarial_prediction)
        attacked_records.append(
            {
                "sample_index": idx,
                "clean_mse": clean_metrics["mse"],
                "adv_mse": adv_metrics["mse"],
                "clean_mae": clean_metrics["mae"],
                "adv_mae": adv_metrics["mae"],
                "clean_whiteness": clean_metrics["whiteness"],
                "adv_whiteness": adv_metrics["whiteness"],
                "clean_objective_value": result.clean_objective_value,
                "adv_objective_value": result.adversarial_objective_value,
                # Legacy aliases retained for backward compatibility with earlier runs.
                "clean_target_loss": result.clean_objective_value,
                "adv_target_loss": result.adversarial_objective_value,
                "query_count": result.query_count,
                "epsilon_budget": result.epsilon_budget,
                "perturbation_l1": result.perturbation_l1,
                "perturbation_l2": result.perturbation_l2,
                "perturbation_linf": result.perturbation_linf,
                "mse_increase": adv_metrics["mse"] - clean_metrics["mse"],
                "mae_increase": adv_metrics["mae"] - clean_metrics["mae"],
            }
        )

    attacked_targets_array = np.stack(attacked_targets)
    attacked_clean_inputs_array = np.stack(attacked_clean_inputs)
    attacked_adv_inputs_array = np.stack(attacked_adv_inputs)
    attacked_clean_predictions_array = np.stack(attacked_clean_predictions)
    attacked_adv_predictions_array = np.stack(attacked_adv_predictions)

    clean_subset_metrics = summarize_regression(attacked_targets_array, attacked_clean_predictions_array)
    attacked_subset_metrics = summarize_regression(attacked_targets_array, attacked_adv_predictions_array)
    per_sample_frame = pd.DataFrame(attacked_records)
    per_sample_frame.to_csv(output_dir / "per_sample_metrics.csv", index=False)

    summary["attack"] = {
        **asdict(config.attack),
        "attacked_samples": attack_count,
        "objective_goal": objective_goal,
        "clean_subset_metrics": clean_subset_metrics,
        "attacked_subset_metrics": attacked_subset_metrics,
        "average_query_count": float(per_sample_frame["query_count"].mean()),
        "average_mse_increase": float(per_sample_frame["mse_increase"].mean()),
        "average_mae_increase": float(per_sample_frame["mae_increase"].mean()),
        "average_clean_objective_value": float(per_sample_frame["clean_objective_value"].mean()),
        "average_adv_objective_value": float(per_sample_frame["adv_objective_value"].mean()),
        # Legacy aliases retained for backward compatibility with earlier analysis.
        "average_clean_target_loss": float(per_sample_frame["clean_objective_value"].mean()),
        "average_adv_target_loss": float(per_sample_frame["adv_objective_value"].mean()),
        "average_whiteness_shift": float(
            (per_sample_frame["adv_whiteness"] - per_sample_frame["clean_whiteness"]).mean()
        ),
    }

    if config.interpretability.enabled:
        summary["interpretability"] = generate_interpretability_outputs(
            config=config.interpretability,
            model=model,
            output_dir=output_dir,
            bundle=AttackInterpretabilityBundle(
                clean_inputs=attacked_clean_inputs_array,
                adversarial_inputs=attacked_adv_inputs_array,
                targets=attacked_targets_array,
                clean_predictions=attacked_clean_predictions_array,
                adversarial_predictions=attacked_adv_predictions_array,
                per_sample_metrics=per_sample_frame,
            ),
        )
    return summary


def _render_summary_markdown(summary: dict[str, object]) -> str:
    experiment = summary["experiment"]
    dataset = summary["dataset"]
    model = summary["model"]
    clean = summary["clean_test_metrics"]
    lines = [
        f"# {experiment['name']}",
        "",
        "## Overview",
        f"- Model: `{model['resolved_name']}`",
        f"- Dataset source: `{dataset['source']}`",
        f"- Features: `{len(dataset['feature_columns'])}`",
        "",
        "## Clean Test Metrics",
        f"- MSE: `{clean['mse']:.6f}`",
        f"- MAE: `{clean['mae']:.6f}`",
        f"- Whiteness: `{clean['whiteness']:.6f}`",
    ]
    if "attack" in summary:
        attack = summary["attack"]
        attacked = attack["attacked_subset_metrics"]
        clean_subset = attack["clean_subset_metrics"]
        lines.extend(
            [
                "",
                "## Attack Subset Metrics",
                f"- Attack objective: `{attack['objective']}`",
                f"- Objective direction: `{attack['objective_goal']}`",
                f"- Clean subset MSE: `{clean_subset['mse']:.6f}`",
                f"- Attacked subset MSE: `{attacked['mse']:.6f}`",
                f"- Clean subset MAE: `{clean_subset['mae']:.6f}`",
                f"- Attacked subset MAE: `{attacked['mae']:.6f}`",
                f"- Avg clean objective value: `{attack['average_clean_objective_value']:.6f}`",
                f"- Avg attacked objective value: `{attack['average_adv_objective_value']:.6f}`",
                f"- Avg query count: `{attack['average_query_count']:.2f}`",
                f"- Avg whiteness shift: `{attack['average_whiteness_shift']:.6f}`",
            ]
        )
    if "interpretability" in summary:
        interpretability = summary["interpretability"]
        horizon = interpretability["horizon"]
        sensitivity = interpretability["input_sensitivity"]
        lines.extend(
            [
                "",
                "## Interpretability",
                f"- Horizon pattern: {horizon['pattern']}",
                f"- Positive attacked-vs-clean MSE steps: `{horizon['positive_mse_steps']}` / `{horizon['total_steps']}`",
                f"- Most sensitive input block: `{sensitivity['most_sensitive_block_index']}`",
                f"- Sensitive context range: `{sensitivity['most_sensitive_context_start']}` to `{sensitivity['most_sensitive_context_end']}`",
                f"- Representative samples: `{interpretability['representative_samples']}`",
                "- Saved interpretability files under `interpretability/`",
            ]
        )
    return "\n".join(lines) + "\n"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
