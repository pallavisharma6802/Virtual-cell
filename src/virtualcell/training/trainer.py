"""Trainer for the conditional delta model: train/val loop, early stopping,
checkpointing, seeded reproducibility. CPU-only, small-data scale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from virtualcell.models.conditional import ConditionalDeltaModel, ConditionalModelConfig
from virtualcell.training.losses import get_loss_fn


@dataclass
class TrainConfig:
    seed: int = 0
    loss: str = "huber"
    lr: float = 1e-3
    max_epochs: int = 200
    patience: int = 15
    weight_decay: float = 1e-5
    batch_size: int | None = None  # None = full-batch (original behavior)


@dataclass
class TrainResult:
    best_epoch: int
    best_val_loss: float
    train_losses: list
    val_losses: list


class Trainer:
    def __init__(self, model_config: ConditionalModelConfig, train_config: TrainConfig):
        self.model_config = model_config
        self.train_config = train_config
        torch.manual_seed(train_config.seed)
        np.random.seed(train_config.seed)
        self.model = ConditionalDeltaModel(model_config)
        self.loss_fn = get_loss_fn(train_config.loss)

    def fit(
        self,
        train_control: np.ndarray,
        train_target_desc: np.ndarray,
        train_delta: np.ndarray,
        val_control: np.ndarray,
        val_target_desc: np.ndarray,
        val_delta: np.ndarray,
    ) -> TrainResult:
        cfg = self.train_config
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        X_ctrl = torch.tensor(train_control, dtype=torch.float32)
        X_tgt = torch.tensor(train_target_desc, dtype=torch.float32)
        y = torch.tensor(train_delta, dtype=torch.float32)

        Xv_ctrl = torch.tensor(val_control, dtype=torch.float32)
        Xv_tgt = torch.tensor(val_target_desc, dtype=torch.float32)
        yv = torch.tensor(val_delta, dtype=torch.float32)

        n_train = X_ctrl.shape[0]
        batch_size = cfg.batch_size if cfg.batch_size is not None else n_train
        rng = np.random.default_rng(cfg.seed)

        best_val_loss = float("inf")
        best_state = None
        best_epoch = -1
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        for epoch in range(cfg.max_epochs):
            self.model.train()
            perm = rng.permutation(n_train) if batch_size < n_train else np.arange(n_train)
            epoch_losses = []
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                pred = self.model(X_ctrl[idx], X_tgt[idx])
                loss = self.loss_fn(pred, y[idx])
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            train_losses.append(float(np.mean(epoch_losses)))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xv_ctrl, Xv_tgt)
                val_loss = self.loss_fn(val_pred, yv)
            val_losses.append(float(val_loss.item()))

            if val_loss.item() < best_val_loss - 1e-6:
                best_val_loss = float(val_loss.item())
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return TrainResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
        )

    def predict_delta(self, control: np.ndarray, target_desc: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(
                torch.tensor(control, dtype=torch.float32),
                torch.tensor(target_desc, dtype=torch.float32),
            )
        return pred.numpy()

    def save(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        config = {
            "model_config": asdict(self.model_config),
            "train_config": asdict(self.train_config),
        }
        (checkpoint_dir / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, checkpoint_dir: Path) -> "Trainer":
        config = json.loads((checkpoint_dir / "config.json").read_text())
        model_config = ConditionalModelConfig(**config["model_config"])
        train_config = TrainConfig(**config["train_config"])
        trainer = cls(model_config, train_config)
        state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
        trainer.model.load_state_dict(state_dict)
        return trainer
