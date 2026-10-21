"""Integration test against the live Arc Virtual Cell Challenge bucket.

Skipped by default: requires network access and the `gcs` optional
dependency group (`pip install -e .[gcs]`). Run explicitly with:
    VIRTUALCELL_RUN_NETWORK_TESTS=1 pytest -q tests/test_arc_adapter.py
"""
import os

import pytest

RUN_NETWORK_TESTS = os.environ.get("VIRTUALCELL_RUN_NETWORK_TESTS") == "1"


@pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason="Set VIRTUALCELL_RUN_NETWORK_TESTS=1 to run this live-bucket integration test.",
)
def test_load_arc_challenge_dataset_schema():
    from virtualcell.data.arc_adapter import ARC_CHALLENGE_TRAIN_URL, load_arc_challenge_dataset

    dataset = load_arc_challenge_dataset(ARC_CHALLENGE_TRAIN_URL, backed="r")

    assert dataset.n_cells == 221_273
    assert dataset.n_genes == 18_080
    assert dataset.provenance.is_synthetic is False
    assert "control" in set(dataset.adata.obs["perturbation_id"].unique())
    assert dataset.adata.obs["is_control"].dtype == bool
