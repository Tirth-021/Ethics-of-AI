from __future__ import annotations

import numpy as np

from llm_ts_audit.attacks.dga import DGAStyleAttack
from llm_ts_audit.config import AttackConfig
from llm_ts_audit.models.base import ForecastModel


class MeanRepeatModel(ForecastModel):
    def __init__(self, horizon: int):
        self.horizon = horizon

    @property
    def name(self) -> str:
        return "mean_repeat"

    def fit(self, x_train, y_train, x_val=None, y_val=None) -> None:
        del x_train, y_train, x_val, y_val

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        mean_value = inputs.mean(axis=1, keepdims=True)
        return np.repeat(mean_value, self.horizon, axis=1).astype(np.float32)


def test_dga_attack_reduces_target_loss_on_simple_model():
    model = MeanRepeatModel(horizon=4)
    clean_input = np.ones((6, 2), dtype=np.float32)
    attack = DGAStyleAttack(
        AttackConfig(
            steps=6,
            directions_per_step=4,
            probe_scale=0.01,
            step_size=0.05,
            epsilon=0.3,
            epsilon_relative_to="absolute",
            norm="linf",
            target_mode="zeros",
            target_source="input",
            loss="mae",
            seed=7,
        )
    )
    result = attack.run(model=model, clean_input=clean_input)
    assert result.adversarial_target_loss < result.clean_target_loss
    assert result.perturbation_linf <= result.epsilon_budget + 1e-6

