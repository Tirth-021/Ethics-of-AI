from __future__ import annotations

from llm_ts_audit.config import ModelConfig
from llm_ts_audit.models.base import ForecastModel
from llm_ts_audit.models.linear import LinearForecaster
from llm_ts_audit.models.torch_models import (
    AutoregressiveTransformerForecaster,
    JointTransformerForecaster,
)


def build_model(
    config: ModelConfig,
    context_length: int,
    horizon: int,
    n_features: int,
) -> ForecastModel:
    params = dict(config.params)
    if config.kind == "linear":
        return LinearForecaster(**params)
    if config.kind == "joint_transformer":
        return JointTransformerForecaster(
            context_length=context_length,
            horizon=horizon,
            n_features=n_features,
            **params,
        )
    if config.kind == "autoregressive_transformer":
        return AutoregressiveTransformerForecaster(
            context_length=context_length,
            horizon=horizon,
            n_features=n_features,
            **params,
        )
    raise ValueError(f"Unsupported model kind: {config.kind}")

