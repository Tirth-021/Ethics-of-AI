from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "debug_experiment"
    seed: int = 7
    output_dir: str = "outputs"


@dataclass
class DatasetConfig:
    source: str = "synthetic"
    path: str | None = None
    time_column: str | None = None
    feature_columns: list[str] | None = None
    context_length: int = 96
    horizon: int = 48
    split: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.25])
    stride: int = 1
    normalize: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    synthetic: dict[str, Any] = field(
        default_factory=lambda: {
            "kind": "sine_spike",
            "length": 1200,
            "n_features": 3,
            "noise_std": 0.05,
        }
    )


@dataclass
class ModelConfig:
    kind: str = "linear"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackConfig:
    enabled: bool = True
    kind: str = "dga"
    objective: str = "target_match"
    max_samples: int = 32
    steps: int = 8
    directions_per_step: int = 4
    probe_scale: float = 0.01
    step_size: float = 0.01
    epsilon: float = 0.02
    epsilon_relative_to: str = "mean_abs_input"
    norm: str = "l1"
    target_mode: str = "gwn"
    target_source: str = "input"
    loss: str = "mae"
    seed: int = 7


@dataclass
class EvaluationConfig:
    prediction_batch_size: int = 128


@dataclass
class InterpretabilityConfig:
    enabled: bool = True
    samples: int = 8
    input_blocks: int = 4
    representative_samples: int = 3
    mask_strategy: str = "input_mean"


@dataclass
class FullConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)


def _merge_dataclass(dataclass_type: type, payload: dict[str, Any] | None):
    payload = payload or {}
    return dataclass_type(**payload)


def load_config(path: str | Path) -> FullConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return FullConfig(
        experiment=_merge_dataclass(ExperimentConfig, raw.get("experiment")),
        dataset=_merge_dataclass(DatasetConfig, raw.get("dataset")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        attack=_merge_dataclass(AttackConfig, raw.get("attack")),
        evaluation=_merge_dataclass(EvaluationConfig, raw.get("evaluation")),
        interpretability=_merge_dataclass(InterpretabilityConfig, raw.get("interpretability")),
    )


def save_config(config: FullConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(asdict(config), sort_keys=False),
        encoding="utf-8",
    )
