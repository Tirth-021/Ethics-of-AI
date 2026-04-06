from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def lag1_autocorrelation(sequence: np.ndarray) -> float:
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


def whiteness_score(sequence: np.ndarray) -> float:
    return float(1.0 - min(1.0, abs(lag1_autocorrelation(sequence))))


def summarize_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "whiteness": whiteness_score(y_pred),
    }

