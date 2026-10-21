import numpy as np
import torch

from virtualcell.models.conditional import ConditionalDeltaModel, ConditionalModelConfig


def _config(**overrides):
    base = dict(n_genes=10, target_descriptor_dim=2)
    base.update(overrides)
    return ConditionalModelConfig(**base)


def test_forward_output_shape_full_model():
    model = ConditionalDeltaModel(_config())
    control = torch.zeros(4, 10)
    target_desc = torch.zeros(4, 2)
    out = model(control, target_desc)
    assert out.shape == (4, 10)


def test_reconstruct_equals_control_plus_delta():
    model = ConditionalDeltaModel(_config())
    control = torch.randn(3, 10)
    target_desc = torch.randn(3, 2)
    delta = model(control, target_desc)
    recon = model.reconstruct(control, target_desc)
    assert torch.allclose(recon, control + delta)


def test_ablation_no_control_context_runs():
    model = ConditionalDeltaModel(_config(use_control_context=False))
    control = torch.randn(2, 10)
    target_desc = torch.randn(2, 2)
    out = model(control, target_desc)
    assert out.shape == (2, 10)


def test_ablation_no_target_descriptor_runs():
    model = ConditionalDeltaModel(_config(use_target_descriptor=False))
    control = torch.randn(2, 10)
    target_desc = torch.randn(2, 2)
    out = model(control, target_desc)
    assert out.shape == (2, 10)


def test_rejects_both_branches_disabled():
    import pytest

    with pytest.raises(ValueError):
        ConditionalDeltaModel(_config(use_control_context=False, use_target_descriptor=False))
