"""Download the full Arc Virtual Cell Challenge training h5ad to local
disk ONCE, as a single sequential transfer.

Root cause found live (2026-11-10, see docs/experiment_log.md): the
sparse X arrays are HDF5-chunked in small (~230KB / ~6-7 rows each)
pieces with no compression. Scattered random-row access (what every
prior scale-up attempt did) means each sampled row typically lands in
its own chunk, so fetching N scattered rows costs ~N separate small
HTTP range requests — latency-dominated and, at larger N, effectively
unbounded. A single sequential whole-file download has none of that
problem: one large linear transfer, then all subsequent row access
(any scatter pattern, any scale) reads from local disk with zero
network latency.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from virtualcell.data.arc_adapter import ARC_CHALLENGE_TRAIN_URL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/adata_Training.h5ad"))
    args = parser.parse_args()

    import gcsfs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ARC_CHALLENGE_TRAIN_URL} -> {args.output} (sequential, ~14.4 GiB)...")
    fs = gcsfs.GCSFileSystem(token="anon")
    t0 = time.time()
    fs.get(ARC_CHALLENGE_TRAIN_URL, str(args.output))
    elapsed = time.time() - t0
    size_gb = args.output.stat().st_size / 1e9
    print(f"Done: {size_gb:.2f} GB in {elapsed:.1f}s ({size_gb * 8 / elapsed:.1f} Gbps)")


if __name__ == "__main__":
    main()
