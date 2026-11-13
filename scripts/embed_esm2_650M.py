"""Re-embed all perturbation genes using ESM2-650M (facebook/esm2_t33_650M_UR50D).

Current cache: data/esm2/esm2_embeddings.npz — 320d from ESM2-8M (esm2_t6_8M_UR50D)
New cache:     data/esm2/esm2_650M_embeddings.npz — 1280d from ESM2-650M

Reuses existing UniProt sequence cache (data/esm2/uniprot_sequences.json) so no
API calls are needed. Only re-runs the transformer forward pass.

Runtime: ~2-4h on CPU (m7i-flex.large) with batch_size=4.
Memory: ~6GB peak for model + activations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer

SEQ_CACHE  = Path("data/esm2/uniprot_sequences.json")
OLD_CACHE  = Path("data/esm2/esm2_embeddings.npz")
NEW_CACHE  = Path("data/esm2/esm2_650M_embeddings.npz")
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 4   # conservative for CPU; increase to 8 if RAM allows
MAX_LEN    = 1022  # ESM2 max is 1024 tokens including BOS/EOS


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def main():
    t0 = time.time()
    NEW_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing sequence cache
    print(f"Loading sequence cache from {SEQ_CACHE} ...")
    with open(SEQ_CACHE) as f:
        seq_cache = json.load(f)
    print(f"  {len(seq_cache)} sequences available")

    # Load old embedding gene list to stay consistent
    old = np.load(str(OLD_CACHE))
    old_genes = old["genes"].tolist()
    print(f"  Old cache has {len(old_genes)} genes (320d)")

    # Filter to genes with sequences
    genes_with_seq = [g for g in old_genes if g in seq_cache and seq_cache[g]]
    missing = [g for g in old_genes if g not in seq_cache or not seq_cache[g]]
    print(f"  {len(genes_with_seq)} genes have sequences; {len(missing)} missing")
    if missing[:5]:
        print(f"  Missing examples: {missing[:5]}")

    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.eval()
    embed_dim = model.config.hidden_size
    print(f"  Model loaded. embed_dim={embed_dim}, params={sum(p.numel() for p in model.parameters()):,}")

    embeddings_dict: dict[str, np.ndarray] = {}
    n = len(genes_with_seq)
    for start in range(0, n, BATCH_SIZE):
        batch_genes = genes_with_seq[start: start + BATCH_SIZE]
        seqs = [seq_cache[g][:MAX_LEN] for g in batch_genes]

        inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN + 2)
        with torch.no_grad():
            out = model(**inputs)
        embs = mean_pool(out.last_hidden_state, inputs["attention_mask"])  # (B, 1280)
        for i, g in enumerate(batch_genes):
            embeddings_dict[g] = embs[i].numpy()

        if (start // BATCH_SIZE) % 25 == 0:
            elapsed = time.time() - t0
            pct = (start + len(batch_genes)) / n * 100
            print(f"  {start + len(batch_genes)}/{n} ({pct:.1f}%) — {elapsed:.0f}s elapsed")

    # For missing genes, fall back to zeros (consistent with old behavior)
    zero = np.zeros(embed_dim, np.float32)
    final_genes, final_embs = [], []
    for g in old_genes:
        final_genes.append(g)
        final_embs.append(embeddings_dict.get(g, zero))

    final_embs_arr = np.array(final_embs, dtype=np.float32)
    np.savez(str(NEW_CACHE), genes=np.array(final_genes), embeddings=final_embs_arr)

    elapsed = time.time() - t0
    print(f"\nSaved {len(final_genes)} embeddings (dim={embed_dim}) to {NEW_CACHE}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Non-zero embeddings: {(final_embs_arr.any(axis=1)).sum()}/{len(final_genes)}")


if __name__ == "__main__":
    main()
