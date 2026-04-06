from __future__ import annotations

import numpy as np

from llm_ts_audit.evaluation.metrics import lag1_autocorrelation, mae, mse, whiteness_score


def test_basic_regression_metrics():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_pred = np.array([[1.0, 1.0], [5.0, 2.0]], dtype=np.float32)
    assert np.isclose(mse(y_true, y_pred), 2.25)
    assert np.isclose(mae(y_true, y_pred), 1.25)


def test_whiteness_score_is_high_for_near_uncorrelated_signal():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=512)
    assert whiteness_score(signal) > 0.8


def test_lag1_autocorrelation_is_high_for_monotonic_signal():
    signal = np.linspace(0.0, 1.0, num=128)
    assert lag1_autocorrelation(signal) > 0.9
