"""Conditional PyTorch model for predicting perturbation-induced delta.

Architecture (plan section 4B, concat+MLP fusion first):
  control_encoder(control_profile) -> control_embedding
  target_encoder(target_descriptor) -> target_embedding
  fusion = MLP(concat(control_embedding, target_embedding))
  delta_head(fusion) -> predicted delta (n_genes)

Ablation support is built in rather than bolted on: `use_control_context`
and `use_target_descriptor` let either branch be zeroed out so the trainer
can run "control removed" / "target removed" ablations with the same
model class (plan section 4B ablation list).

Target descriptor caveat (plan section 4A "Unseen-target caveat"): no
pretrained gene-representation model is wired in here. The default
descriptor (`control_mean_std`) uses only each target gene's own
control-population mean and std — a real, generalizable, non-leaking
feature that exists for any target gene (including held-out ones), but it
is a weak hand-built descriptor, not a rich pretrained embedding. This is
documented, not disguised: see docs/experiment_log.md Milestone 3 entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ConditionalModelConfig:
    n_genes: int
    target_descriptor_dim: int
    control_hidden_dim: int = 128
    target_hidden_dim: int = 32
    fusion_hidden_dim: int = 128
    use_control_context: bool = True
    use_target_descriptor: bool = True


class ControlEncoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, control_profile: torch.Tensor) -> torch.Tensor:
        return self.net(control_profile)


class TargetEncoder(nn.Module):
    def __init__(self, descriptor_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, target_descriptor: torch.Tensor) -> torch.Tensor:
        return self.net(target_descriptor)


class ConditionalDeltaModel(nn.Module):
    """Predicts gene-wise expression delta from control state + target info."""

    def __init__(self, config: ConditionalModelConfig):
        super().__init__()
        self.config = config
        self.control_encoder = ControlEncoder(config.n_genes, config.control_hidden_dim)
        self.target_encoder = TargetEncoder(config.target_descriptor_dim, config.target_hidden_dim)

        fusion_input_dim = 0
        if config.use_control_context:
            fusion_input_dim += config.control_hidden_dim
        if config.use_target_descriptor:
            fusion_input_dim += config.target_hidden_dim
        if fusion_input_dim == 0:
            raise ValueError("At least one of control context or target descriptor must be enabled")

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(config.fusion_hidden_dim, config.n_genes)

    def forward(self, control_profile: torch.Tensor, target_descriptor: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.config.use_control_context:
            parts.append(self.control_encoder(control_profile))
        if self.config.use_target_descriptor:
            parts.append(self.target_encoder(target_descriptor))
        fused = self.fusion(torch.cat(parts, dim=-1))
        return self.delta_head(fused)

    def reconstruct(self, control_profile: torch.Tensor, target_descriptor: torch.Tensor) -> torch.Tensor:
        """Predicted post-perturbation expression = control + predicted delta,
        in the same numeric space as the input control profile."""
        return control_profile + self.forward(control_profile, target_descriptor)
