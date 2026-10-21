"""Loss functions for delta prediction."""
from __future__ import annotations

import torch
from torch import nn


def get_loss_fn(name: str) -> nn.Module:
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")
