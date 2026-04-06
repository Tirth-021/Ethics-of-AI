from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from llm_ts_audit.models.base import ForecastModel


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for transformer models. Install with `pip install -e \".[torch]\"`."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


@dataclass
class TorchTrainConfig:
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    device: str | None = None


class _PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 2048):
        torch, _, _, _ = _require_torch()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def to(self, device):
        self.pe = self.pe.to(device)
        return self

    def __call__(self, x):
        return x + self.pe[:, : x.size(1)]


class _BaseTorchForecaster(ForecastModel):
    def __init__(self, context_length: int, horizon: int, n_features: int, **kwargs):
        self.context_length = context_length
        self.horizon = horizon
        self.n_features = n_features
        self.train_config = TorchTrainConfig(**kwargs)
        self.model = None
        self.device = None
        self._torch = None

    def _prepare_runtime(self):
        torch, nn, DataLoader, TensorDataset = _require_torch()
        self._torch = torch
        if self.train_config.device:
            self.device = self.train_config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            self.model = self._build_network(nn).to(self.device)
        return torch, nn, DataLoader, TensorDataset

    def _build_network(self, nn):
        raise NotImplementedError

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        torch, nn, DataLoader, TensorDataset = self._prepare_runtime()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()),
            batch_size=self.train_config.batch_size,
            shuffle=True,
        )
        val_loader = None
        if x_val is not None and y_val is not None:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()),
                batch_size=self.train_config.batch_size,
                shuffle=False,
            )

        best_state = None
        best_val = float("inf")
        patience_left = self.train_config.patience

        for _epoch in range(self.train_config.epochs):
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                prediction = self.model(batch_x)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()

            current_val = self._evaluate_loss(val_loader, loss_fn)
            if current_val < best_val:
                best_val = current_val
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience_left = self.train_config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def _evaluate_loss(self, val_loader, loss_fn) -> float:
        if val_loader is None:
            return 0.0
        total_loss = 0.0
        total_items = 0
        self.model.eval()
        with self._torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                prediction = self.model(batch_x)
                total_loss += float(loss_fn(prediction, batch_y).item()) * len(batch_x)
                total_items += len(batch_x)
        return total_loss / max(total_items, 1)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        torch = self._torch
        self.model.eval()
        batch_size = self.train_config.batch_size
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = torch.from_numpy(inputs[start : start + batch_size]).float().to(self.device)
                outputs.append(self.model(batch).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


class JointTransformerForecaster(_BaseTorchForecaster):
    @property
    def name(self) -> str:
        return "joint_transformer"

    def _build_network(self, nn):
        cfg = self.train_config
        pos_encoding = _PositionalEncoding(cfg.d_model, max_len=self.context_length + 8)

        class Network(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
                self.input_proj = nn.Linear(outer.n_features, cfg.d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.ff_dim,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
                self.norm = nn.LayerNorm(cfg.d_model)
                self.head = nn.Sequential(
                    nn.Linear(cfg.d_model, cfg.ff_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.ff_dim, outer.horizon * outer.n_features),
                )
                self.pos_encoding = pos_encoding

            def forward(self, x):
                x = self.input_proj(x)
                x = self.pos_encoding.to(x.device)(x)
                encoded = self.encoder(x)
                pooled = self.norm(encoded.mean(dim=1))
                out = self.head(pooled)
                return out.view(-1, self.outer.horizon, self.outer.n_features)

        return Network(self)


class AutoregressiveTransformerForecaster(_BaseTorchForecaster):
    @property
    def name(self) -> str:
        return "autoregressive_transformer"

    def _build_network(self, nn):
        cfg = self.train_config
        pos_encoding = _PositionalEncoding(cfg.d_model, max_len=self.context_length + 8)

        class Network(nn.Module):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer
                self.input_proj = nn.Linear(outer.n_features, cfg.d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.ff_dim,
                    dropout=cfg.dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
                self.state_proj = nn.Linear(cfg.d_model, cfg.ff_dim)
                self.decoder = nn.GRUCell(outer.n_features, cfg.ff_dim)
                self.head = nn.Linear(cfg.ff_dim, outer.n_features)
                self.pos_encoding = pos_encoding

            def _causal_mask(self, length, device):
                return self.outer._torch.triu(
                    self.outer._torch.ones(length, length, device=device) * float("-inf"),
                    diagonal=1,
                )

            def forward(self, x):
                encoded = self.input_proj(x)
                encoded = self.pos_encoding.to(x.device)(encoded)
                encoded = self.encoder(encoded, mask=self._causal_mask(x.size(1), x.device))
                hidden = self.state_proj(encoded[:, -1, :])
                current = x[:, -1, :]
                outputs = []
                for _ in range(self.outer.horizon):
                    hidden = self.decoder(current, hidden)
                    current = self.head(hidden)
                    outputs.append(current.unsqueeze(1))
                return self.outer._torch.cat(outputs, dim=1)

        return Network(self)

