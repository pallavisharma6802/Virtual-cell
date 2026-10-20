import numpy as np

from virtualcell.data.adapters import from_anndata
from virtualcell.data.fixtures import make_synthetic_dataset
from virtualcell.data.splits import held_out_target_split
from virtualcell.models.baselines import NoEffectBaseline, SeenTargetMeanBaseline


def test_no_effect_baseline_predicts_zero():
    baseline = NoEffectBaseline(n_genes=10)
    pred = baseline.predict_delta(n_cells=5)
    assert pred.shape == (5, 10)
    assert np.all(pred == 0)


def test_seen_target_mean_baseline_has_no_overlap_with_held_out_targets():
    adata = make_synthetic_dataset(seed=4)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, seed=0)

    baseline = SeenTargetMeanBaseline().fit(dataset, manifest)

    known = baseline.known_targets()
    assert known <= set(manifest.train_targets)
    assert known & set(manifest.test_targets) == set()


def test_seen_target_mean_baseline_unseen_target_falls_back_to_zero_delta():
    adata = make_synthetic_dataset(seed=4)
    dataset = from_anndata(adata, source_name="synthetic_fixture", is_synthetic=True)
    manifest = held_out_target_split(dataset, seed=0)
    baseline = SeenTargetMeanBaseline().fit(dataset, manifest)

    unseen_target = manifest.test_targets[0]
    pred = baseline.predict_delta([unseen_target])
    assert np.all(pred == 0)
