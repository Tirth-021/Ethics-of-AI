from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from llm_ts_audit.models.base import ForecastModel


class LinearForecaster(ForecastModel):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self._horizon: int | None = None
        self._n_features: int | None = None

    @property
    def name(self) -> str:
        return "linear_ridge"

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        del x_val, y_val
        n_samples, _, n_features = x_train.shape
        self._horizon = y_train.shape[1]
        self._n_features = n_features
        self.model.fit(
            x_train.reshape(n_samples, -1),
            y_train.reshape(n_samples, -1),
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self._horizon is None or self._n_features is None:
            raise RuntimeError("Model must be fitted before prediction.")
        n_samples = inputs.shape[0]
        outputs = self.model.predict(inputs.reshape(n_samples, -1))
        return outputs.reshape(n_samples, self._horizon, self._n_features).astype(np.float32)

