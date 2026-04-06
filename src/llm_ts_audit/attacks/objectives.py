from __future__ import annotations

import numpy as np


def objective_direction(objective: str) -> str:
    if objective == "target_match":
        return "minimize"
    if objective in {"clean_deviation", "oracle_error", "whiteness_max"}:
        return "maximize"
    raise ValueError(f"Unsupported attack objective: {objective}")


def build_target_sequence(
    mode: str,
    reference_input: np.ndarray,
    clean_prediction: np.ndarray,
    rng: np.random.Generator,
    source: str = "input",
) -> np.ndarray:
    if mode == "zeros":
        return np.zeros_like(clean_prediction, dtype=np.float32)

    source_array = clean_prediction if source == "prediction" else reference_input
    mean = np.mean(source_array, axis=0, keepdims=True)
    std = np.std(source_array, axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)

    if mode == "gwn":
        target = rng.normal(
            loc=np.broadcast_to(mean, clean_prediction.shape),
            scale=np.broadcast_to(std, clean_prediction.shape),
        )
        return target.astype(np.float32)

    if mode == "mirror_clean":
        return clean_prediction[::-1].astype(np.float32)

    raise ValueError(f"Unsupported target mode: {mode}")


def target_loss(prediction: np.ndarray, target: np.ndarray, loss_kind: str = "mae") -> float:
    if loss_kind == "mse":
        return float(np.mean((prediction - target) ** 2))
    if loss_kind == "mae":
        return float(np.mean(np.abs(prediction - target)))
    raise ValueError(f"Unsupported loss kind: {loss_kind}")


def evaluate_attack_objective(
    objective: str,
    prediction: np.ndarray,
    loss_kind: str,
    clean_prediction: np.ndarray | None = None,
    target: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
) -> float:
    if objective == "target_match":
        if target is None:
            raise ValueError("target_match objective requires a target sequence.")
        return target_loss(prediction, target, loss_kind)

    if objective == "clean_deviation":
        if clean_prediction is None:
            raise ValueError("clean_deviation objective requires clean_prediction.")
        return target_loss(prediction, clean_prediction, loss_kind)

    if objective == "oracle_error":
        if ground_truth is None:
            raise ValueError("oracle_error objective requires ground_truth.")
        return target_loss(prediction, ground_truth, loss_kind)

    if objective == "whiteness_max":
        return whiteness_score(prediction)

    raise ValueError(f"Unsupported attack objective: {objective}")


def whiteness_score(sequence: np.ndarray) -> float:
    return float(1.0 - min(1.0, abs(_lag1_autocorrelation(sequence))))


def _lag1_autocorrelation(sequence: np.ndarray) -> float:
    flat = np.asarray(sequence, dtype=np.float64).reshape(-1)
    if flat.size < 2:
        return 0.0
    left = flat[:-1]
    right = flat[1:]
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denom)
