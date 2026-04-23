from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from llm_ts_audit.config import InterpretabilityConfig
from llm_ts_audit.evaluation.metrics import mae, mse
from llm_ts_audit.models.base import ForecastModel


@dataclass
class AttackInterpretabilityBundle:
    clean_inputs: np.ndarray
    adversarial_inputs: np.ndarray
    targets: np.ndarray
    clean_predictions: np.ndarray
    adversarial_predictions: np.ndarray
    per_sample_metrics: pd.DataFrame


def generate_interpretability_outputs(
    config: InterpretabilityConfig,
    model: ForecastModel,
    output_dir: Path,
    bundle: AttackInterpretabilityBundle,
) -> dict[str, object]:
    interpretability_dir = output_dir / "interpretability"
    interpretability_dir.mkdir(parents=True, exist_ok=True)

    horizon_profile = _build_horizon_profile(
        targets=bundle.targets,
        clean_predictions=bundle.clean_predictions,
        adversarial_predictions=bundle.adversarial_predictions,
    )
    horizon_profile.to_csv(interpretability_dir / "horizon_error_profile.csv", index=False)

    sensitivity_records, sensitivity_summary = _build_input_block_sensitivity(
        config=config,
        model=model,
        clean_inputs=bundle.clean_inputs,
        targets=bundle.targets,
        clean_predictions=bundle.clean_predictions,
    )
    sensitivity_records.to_csv(interpretability_dir / "input_block_sensitivity.csv", index=False)
    sensitivity_summary.to_csv(interpretability_dir / "input_block_sensitivity_summary.csv", index=False)

    representative_forecasts = _build_representative_forecasts(
        config=config,
        bundle=bundle,
    )
    representative_forecasts.to_csv(interpretability_dir / "representative_forecasts.csv", index=False)

    summary = _build_interpretability_summary(
        config=config,
        horizon_profile=horizon_profile,
        sensitivity_summary=sensitivity_summary,
        per_sample_metrics=bundle.per_sample_metrics,
    )
    (interpretability_dir / "summary.md").write_text(
        _render_interpretability_markdown(summary),
        encoding="utf-8",
    )
    return summary


def _build_horizon_profile(
    targets: np.ndarray,
    clean_predictions: np.ndarray,
    adversarial_predictions: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    horizon = targets.shape[1]
    for step_idx in range(horizon):
        y_true_step = targets[:, step_idx, :]
        clean_step = clean_predictions[:, step_idx, :]
        adv_step = adversarial_predictions[:, step_idx, :]
        clean_mse_value = mse(y_true_step, clean_step)
        adv_mse_value = mse(y_true_step, adv_step)
        clean_mae_value = mae(y_true_step, clean_step)
        adv_mae_value = mae(y_true_step, adv_step)
        records.append(
            {
                "horizon_step": step_idx + 1,
                "clean_mse": clean_mse_value,
                "adv_mse": adv_mse_value,
                "mse_increase": adv_mse_value - clean_mse_value,
                "clean_mae": clean_mae_value,
                "adv_mae": adv_mae_value,
                "mae_increase": adv_mae_value - clean_mae_value,
            }
        )
    return pd.DataFrame(records)


def _build_input_block_sensitivity(
    config: InterpretabilityConfig,
    model: ForecastModel,
    clean_inputs: np.ndarray,
    targets: np.ndarray,
    clean_predictions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_count = min(config.samples, len(clean_inputs))
    block_count = max(config.input_blocks, 1)
    block_indices = np.array_split(np.arange(clean_inputs.shape[1]), block_count)
    records: list[dict[str, float | int]] = []

    for sample_idx in range(sample_count):
        clean_input = clean_inputs[sample_idx]
        target = targets[sample_idx]
        baseline_prediction = clean_predictions[sample_idx]
        baseline_mse = mse(target, baseline_prediction)
        baseline_mae = mae(target, baseline_prediction)
        replacement = _mask_reference(clean_input, config.mask_strategy)

        for block_idx, positions in enumerate(block_indices):
            masked_input = clean_input.copy()
            masked_input[positions] = replacement
            masked_prediction = model.predict(masked_input[None, ...])[0]
            records.append(
                {
                    "sample_index": sample_idx,
                    "block_index": block_idx,
                    "context_start": int(positions[0]) + 1,
                    "context_end": int(positions[-1]) + 1,
                    "forecast_shift_mae": float(np.mean(np.abs(masked_prediction - baseline_prediction))),
                    "mse_increase": mse(target, masked_prediction) - baseline_mse,
                    "mae_increase": mae(target, masked_prediction) - baseline_mae,
                }
            )

    detail_frame = pd.DataFrame(records)
    summary_frame = (
        detail_frame.groupby("block_index", as_index=False)
        .agg(
            context_start=("context_start", "first"),
            context_end=("context_end", "first"),
            average_forecast_shift_mae=("forecast_shift_mae", "mean"),
            average_mse_increase=("mse_increase", "mean"),
            average_mae_increase=("mae_increase", "mean"),
        )
        .sort_values("block_index")
        .reset_index(drop=True)
    )
    return detail_frame, summary_frame


def _build_representative_forecasts(
    config: InterpretabilityConfig,
    bundle: AttackInterpretabilityBundle,
) -> pd.DataFrame:
    top_indices = (
        bundle.per_sample_metrics.sort_values("mse_increase", ascending=False)["sample_index"]
        .head(config.representative_samples)
        .tolist()
    )
    records: list[dict[str, float | int | str]] = []

    for sample_index in top_indices:
        clean_input = bundle.clean_inputs[sample_index]
        adversarial_input = bundle.adversarial_inputs[sample_index]
        target = bundle.targets[sample_index]
        clean_prediction = bundle.clean_predictions[sample_index]
        adv_prediction = bundle.adversarial_predictions[sample_index]

        for context_idx in range(clean_input.shape[0]):
            for feature_idx in range(clean_input.shape[1]):
                records.append(
                    {
                        "sample_index": sample_index,
                        "segment": "context",
                        "time_step": context_idx + 1,
                        "feature_index": feature_idx,
                        "series": "clean_input",
                        "value": float(clean_input[context_idx, feature_idx]),
                    }
                )
                records.append(
                    {
                        "sample_index": sample_index,
                        "segment": "context",
                        "time_step": context_idx + 1,
                        "feature_index": feature_idx,
                        "series": "adversarial_input",
                        "value": float(adversarial_input[context_idx, feature_idx]),
                    }
                )

        for horizon_idx in range(target.shape[0]):
            for feature_idx in range(target.shape[1]):
                records.append(
                    {
                        "sample_index": sample_index,
                        "segment": "forecast",
                        "time_step": horizon_idx + 1,
                        "feature_index": feature_idx,
                        "series": "ground_truth",
                        "value": float(target[horizon_idx, feature_idx]),
                    }
                )
                records.append(
                    {
                        "sample_index": sample_index,
                        "segment": "forecast",
                        "time_step": horizon_idx + 1,
                        "feature_index": feature_idx,
                        "series": "clean_prediction",
                        "value": float(clean_prediction[horizon_idx, feature_idx]),
                    }
                )
                records.append(
                    {
                        "sample_index": sample_index,
                        "segment": "forecast",
                        "time_step": horizon_idx + 1,
                        "feature_index": feature_idx,
                        "series": "adversarial_prediction",
                        "value": float(adv_prediction[horizon_idx, feature_idx]),
                    }
                )

    return pd.DataFrame(records)


def _build_interpretability_summary(
    config: InterpretabilityConfig,
    horizon_profile: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    per_sample_metrics: pd.DataFrame,
) -> dict[str, object]:
    horizon_len = len(horizon_profile)
    third = max(horizon_len // 3, 1)
    early = horizon_profile.iloc[:third]["mse_increase"].mean()
    middle = horizon_profile.iloc[third : min(2 * third, horizon_len)]["mse_increase"].mean()
    late = horizon_profile.iloc[min(2 * third, horizon_len) :]["mse_increase"].mean()
    positive_steps = int((horizon_profile["mse_increase"] > 0).sum())
    most_sensitive_block = sensitivity_summary.sort_values(
        "average_forecast_shift_mae",
        ascending=False,
    ).iloc[0]
    representative_samples = (
        per_sample_metrics.sort_values("mse_increase", ascending=False)["sample_index"]
        .head(config.representative_samples)
        .tolist()
    )

    if late > early * 1.2:
        horizon_pattern = "later forecast steps are more sensitive than earlier steps"
    elif early > late * 1.2:
        horizon_pattern = "earlier forecast steps are more sensitive than later steps"
    else:
        horizon_pattern = "error increases are spread broadly across the forecast horizon"

    return {
        "horizon": {
            "positive_mse_steps": positive_steps,
            "total_steps": horizon_len,
            "average_early_mse_increase": float(early),
            "average_middle_mse_increase": float(middle),
            "average_late_mse_increase": float(late),
            "pattern": horizon_pattern,
        },
        "input_sensitivity": {
            "most_sensitive_block_index": int(most_sensitive_block["block_index"]),
            "most_sensitive_context_start": int(most_sensitive_block["context_start"]),
            "most_sensitive_context_end": int(most_sensitive_block["context_end"]),
            "most_sensitive_block_average_shift": float(most_sensitive_block["average_forecast_shift_mae"]),
        },
        "representative_samples": representative_samples,
        "mask_strategy": config.mask_strategy,
        "files": {
            "horizon_profile": "interpretability/horizon_error_profile.csv",
            "input_block_sensitivity": "interpretability/input_block_sensitivity_summary.csv",
            "representative_forecasts": "interpretability/representative_forecasts.csv",
        },
    }


def _mask_reference(clean_input: np.ndarray, mask_strategy: str) -> np.ndarray:
    if mask_strategy == "zeros":
        return np.zeros((1, clean_input.shape[1]), dtype=clean_input.dtype)
    if mask_strategy == "input_mean":
        return clean_input.mean(axis=0, keepdims=True).astype(clean_input.dtype)
    raise ValueError(f"Unsupported interpretability mask strategy: {mask_strategy}")


def _render_interpretability_markdown(summary: dict[str, object]) -> str:
    horizon = summary["horizon"]
    sensitivity = summary["input_sensitivity"]
    files = summary["files"]
    lines = [
        "# Interpretability Summary",
        "",
        "## Horizon-Wise Error Profile",
        f"- Positive attacked-vs-clean MSE steps: `{horizon['positive_mse_steps']}` / `{horizon['total_steps']}`",
        f"- Early-horizon average MSE increase: `{horizon['average_early_mse_increase']:.6f}`",
        f"- Middle-horizon average MSE increase: `{horizon['average_middle_mse_increase']:.6f}`",
        f"- Late-horizon average MSE increase: `{horizon['average_late_mse_increase']:.6f}`",
        f"- Interpretation: {horizon['pattern']}.",
        "",
        "## Input-Block Sensitivity",
        f"- Mask strategy: `{summary['mask_strategy']}`",
        f"- Most sensitive input block: `{sensitivity['most_sensitive_block_index']}`",
        f"- Context range: `{sensitivity['most_sensitive_context_start']}` to `{sensitivity['most_sensitive_context_end']}`",
        f"- Average forecast shift for that block: `{sensitivity['most_sensitive_block_average_shift']:.6f}`",
        "",
        "## Representative Forecast Samples",
        f"- Highest-impact sample indices: `{summary['representative_samples']}`",
        "",
        "## Saved Files",
        f"- Horizon profile: `{files['horizon_profile']}`",
        f"- Input sensitivity summary: `{files['input_block_sensitivity']}`",
        f"- Representative traces: `{files['representative_forecasts']}`",
    ]
    return "\n".join(lines) + "\n"

