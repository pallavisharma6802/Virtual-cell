import numpy as np

from virtualcell.models.conditional import ConditionalModelConfig
from virtualcell.training.trainer import TrainConfig, Trainer


def _toy_data(n=20, n_genes=6, descriptor_dim=2, seed=0):
    rng = np.random.default_rng(seed)
    control = rng.normal(size=(n, n_genes)).astype(np.float32)
    target_desc = rng.normal(size=(n, descriptor_dim)).astype(np.float32)
    true_w = rng.normal(size=(descriptor_dim, n_genes)).astype(np.float32)
    delta = (target_desc @ true_w).astype(np.float32)
    return control, target_desc, delta


def test_trainer_fits_and_reduces_val_loss():
    train_ctrl, train_desc, train_delta = _toy_data(n=40, seed=1)
    val_ctrl, val_desc, val_delta = _toy_data(n=10, seed=2)

    model_config = ConditionalModelConfig(n_genes=6, target_descriptor_dim=2)
    train_config = TrainConfig(seed=0, max_epochs=100, patience=20)
    trainer = Trainer(model_config, train_config)
    result = trainer.fit(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta)

    assert result.val_losses[-1] <= result.val_losses[0]
    assert result.best_val_loss <= result.val_losses[0]


def test_trainer_save_and_load_roundtrip(tmp_path):
    train_ctrl, train_desc, train_delta = _toy_data(n=30, seed=1)
    val_ctrl, val_desc, val_delta = _toy_data(n=8, seed=2)

    model_config = ConditionalModelConfig(n_genes=6, target_descriptor_dim=2)
    train_config = TrainConfig(seed=0, max_epochs=30, patience=10)
    trainer = Trainer(model_config, train_config)
    trainer.fit(train_ctrl, train_desc, train_delta, val_ctrl, val_desc, val_delta)

    pred_before = trainer.predict_delta(val_ctrl, val_desc)
    trainer.save(tmp_path / "ckpt")
    reloaded = Trainer.load(tmp_path / "ckpt")
    pred_after = reloaded.predict_delta(val_ctrl, val_desc)

    assert np.allclose(pred_before, pred_after, atol=1e-5)
