"""Fetch DepMap CRISPR gene effect scores for K562.

DepMap measures CRISPR knockout essentiality across cancer cell lines.
K562 scores directly measure how each gene's loss affects K562 cells —
the same cells used in the Replogle Perturb-seq dataset.

We download the CERES gene effect matrix and extract K562 scores.
These are real-valued scores: negative = essential (cell dies), 0 = no effect.

Output: data/depmap/k562_gene_effect.json — {gene_name: score}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/depmap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CRISPR_FILE = OUT_DIR / "CRISPRGeneEffect.csv"

if not CRISPR_FILE.exists():
    print("ERROR: CRISPRGeneEffect.csv not found.")
    print("Download from: https://depmap.org/portal/download/")
    print("  File: CRISPRGeneEffect.csv (DepMap Public release)")
    print(f"  Place at: {CRISPR_FILE.absolute()}")
    sys.exit(1)

print(f"Loading {CRISPR_FILE} ...")
df = pd.read_csv(CRISPR_FILE, index_col=0)
print(f"  Shape: {df.shape} (cell_lines x genes)")
print(f"  Columns (sample): {list(df.columns[:5])}")

# Find K562 row (cell line names vary by release; look for substring match)
k562_rows = [idx for idx in df.index if "K562" in str(idx)]
print(f"  K562 rows found: {k562_rows}")

if not k562_rows:
    print("ERROR: No K562 row found. Available rows (first 20):")
    print(list(df.index[:20]))
    sys.exit(1)

k562_row = df.loc[k562_rows[0]]
print(f"  Using K562 row: {k562_rows[0]}, {k562_row.notna().sum()} non-null values")

# Column format is "GENE (ENTREZ_ID)" — extract gene name
gene_effect = {}
for col, val in k562_row.items():
    if pd.isna(val):
        continue
    gene_name = col.split(" (")[0].strip()
    gene_effect[gene_name] = float(val)

print(f"  Extracted {len(gene_effect)} gene scores")
print(f"  Score range: {min(gene_effect.values()):.3f} to {max(gene_effect.values()):.3f}")
print(f"  Median: {np.median(list(gene_effect.values())):.4f}")

out_path = OUT_DIR / "k562_gene_effect.json"
with open(out_path, "w") as f:
    json.dump(gene_effect, f)
print(f"Saved: {out_path}")
