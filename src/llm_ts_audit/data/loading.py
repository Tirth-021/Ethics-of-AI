from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from llm_ts_audit.config import DatasetConfig


@dataclass
class WindowSplit:
    inputs: np.ndarray
    targets: np.ndarray


@dataclass
class PreparedDataset:
    train: WindowSplit
    val: WindowSplit
    test: WindowSplit
    feature_columns: list[str]
    scaler: StandardScaler | None
    metadata: dict[str, float | int | str]


def load_prepared_dataset(config: DatasetConfig, seed: int) -> PreparedDataset:
    dataframe, feature_columns = _load_dataframe(config, seed)
    values = dataframe[feature_columns].to_numpy(dtype=np.float64)
    train_ratio, val_ratio, test_ratio = config.split
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Dataset split ratios must sum to 1.0.")

    n_rows = len(values)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    if train_end <= config.context_length + config.horizon:
        raise ValueError("Train split is too small for the requested context/horizon.")

    scaler = None
    if config.normalize:
        scaler = StandardScaler()
        scaler.fit(values[:train_end])
        values = scaler.transform(values)

    train_values = values[:train_end]
    val_values = values[max(0, train_end - config.context_length) : val_end]
    test_values = values[max(0, val_end - config.context_length) :]

    train = _make_windows(
        train_values,
        config.context_length,
        config.horizon,
        stride=config.stride,
        max_samples=config.max_train_samples,
    )
    val = _make_windows(
        val_values,
        config.context_length,
        config.horizon,
        stride=config.stride,
        max_samples=config.max_eval_samples,
    )
    test = _make_windows(
        test_values,
        config.context_length,
        config.horizon,
        stride=config.stride,
        max_samples=config.max_eval_samples,
    )

    metadata = {
        "n_rows": n_rows,
        "n_features": len(feature_columns),
        "train_windows": int(len(train.inputs)),
        "val_windows": int(len(val.inputs)),
        "test_windows": int(len(test.inputs)),
    }
    return PreparedDataset(
        train=train,
        val=val,
        test=test,
        feature_columns=feature_columns,
        scaler=scaler,
        metadata=metadata,
    )


def _load_dataframe(config: DatasetConfig, seed: int) -> tuple[pd.DataFrame, list[str]]:
    if config.source == "synthetic":
        dataframe = _generate_synthetic_dataframe(config.synthetic, seed)
    elif config.path:
        dataframe = pd.read_csv(Path(config.path))
    else:
        raise ValueError("dataset.path must be provided for non-synthetic sources.")

    time_column = config.time_column
    if time_column and time_column not in dataframe.columns:
        time_column = None

    if config.feature_columns:
        missing = [column for column in config.feature_columns if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        feature_columns = config.feature_columns
    else:
        feature_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        if time_column in feature_columns:
            feature_columns.remove(time_column)
        for common_time_name in ("timestamp", "date", "datetime"):
            if common_time_name in feature_columns:
                feature_columns.remove(common_time_name)

    if not feature_columns:
        raise ValueError("No numeric feature columns available for forecasting.")
    return dataframe, feature_columns


def _generate_synthetic_dataframe(settings: dict[str, object], seed: int) -> pd.DataFrame:
    kind = str(settings.get("kind", "sine_spike"))
    length = int(settings.get("length", 1200))
    n_features = int(settings.get("n_features", 3))
    noise_std = float(settings.get("noise_std", 0.05))
    rng = np.random.default_rng(seed)
    time_index = np.arange(length, dtype=np.float64)

    data: dict[str, np.ndarray] = {}
    for feature_idx in range(n_features):
        base_period = 24.0 + (feature_idx * 9.0)
        long_period = 96.0 + (feature_idx * 13.0)
        trend = 0.002 * time_index * ((feature_idx + 1) / n_features)
        series = np.sin(2 * np.pi * time_index / base_period)
        series += 0.6 * np.sin(2 * np.pi * time_index / long_period)
        series += trend
        if kind == "sine_spike":
            spike_center = length // 2 + feature_idx * 7
            if spike_center < length:
                series[spike_center : min(length, spike_center + 5)] += 1.5
        series += rng.normal(0.0, noise_std, size=length)
        data[f"feature_{feature_idx}"] = series

    frame = pd.DataFrame(data)
    frame.insert(0, "timestamp", np.arange(length))
    return frame


def _make_windows(
    values: np.ndarray,
    context_length: int,
    horizon: int,
    stride: int,
    max_samples: int | None,
) -> WindowSplit:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    upper_bound = len(values) - context_length - horizon + 1
    for start in range(0, max(0, upper_bound), stride):
        context = values[start : start + context_length]
        target = values[start + context_length : start + context_length + horizon]
        inputs.append(context)
        targets.append(target)
        if max_samples is not None and len(inputs) >= max_samples:
            break

    if not inputs:
        raise ValueError(
            "Unable to generate any sliding windows. Increase the data size or reduce "
            "context_length/horizon."
        )

    return WindowSplit(
        inputs=np.stack(inputs).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
    )
