"""Standalone worker: fetches ONE chunk of cells from the Arc Virtual
Cell Challenge GCS file in its own OS process, and saves the dense
result to a .npy file.

Why a separate process, not a thread: repeated in-process attempts
(even in a fresh thread) were observed to still fail after gcsfs's
background async prefetcher failed to tear down cleanly once
("BackgroundPrefetcher teardown did not complete... it will keep
running in the background") — the corrupted async/event-loop state
appears to persist for the rest of that Python process, so later
"fresh" attempts in the same process were not actually independent.
A genuinely separate process gives a fresh interpreter and event loop
every time, and if it hangs, the parent can just kill the whole process
with no leftover state to worry about. See docs/experiment_log.md.

Usage: python _fetch_chunk_worker.py <path_or_url> <positions_npy> <output_npy> [top_gene_idx_npy]
"""
from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    path_or_url = sys.argv[1]
    positions_path = sys.argv[2]
    output_path = sys.argv[3]
    top_gene_idx_path = sys.argv[4] if len(sys.argv) > 4 else None

    from virtualcell.data.arc_adapter import load_arc_challenge_dataset

    positions = np.load(positions_path).tolist()
    fresh = load_arc_challenge_dataset(path_or_url, backed="r")
    sub_chunk = fresh.adata[positions, :].to_memory()
    chunk_dense = np.asarray(sub_chunk.X.todense()) if hasattr(sub_chunk.X, "todense") else np.asarray(sub_chunk.X)

    if top_gene_idx_path:
        top_gene_idx = np.load(top_gene_idx_path)
        chunk_dense = chunk_dense[:, top_gene_idx]

    np.save(output_path, chunk_dense.astype(np.float32))


if __name__ == "__main__":
    main()
