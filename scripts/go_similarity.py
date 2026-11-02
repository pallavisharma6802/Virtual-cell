"""GEARS-lite building block: gene-gene functional similarity from GO
Biological Process annotations, used to retrieve/propagate perturbation
effects between functionally related genes instead of describing a
gene's identity as a flat feature vector fed to a generic regressor.

Source: geneontology.org current release GAF file (goa_human.gaf.gz,
~15MB), https://current.geneontology.org/annotations/goa_human.gaf.gz.
Column layout: https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/
"""
from __future__ import annotations

import gzip
from pathlib import Path


def load_go_bp_annotations(gaf_path: str | Path) -> dict[str, set[str]]:
    """Parse a GAF file into {gene_symbol: {GO:id, ...}}, restricted to
    the Biological Process aspect (column index 8 == 'P'), which is the
    aspect GEARS itself uses for its gene similarity graph."""
    gene_to_go: dict[str, set[str]] = {}
    opener = gzip.open if str(gaf_path).endswith(".gz") else open
    with opener(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            symbol, go_id, aspect = cols[2], cols[4], cols[8]
            if aspect != "P":
                continue
            gene_to_go.setdefault(symbol, set()).add(go_id)
    return gene_to_go


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def top_k_neighbors(
    query_gene: str,
    candidate_genes: list[str],
    gene_to_go: dict[str, set[str]],
    k: int,
) -> list[tuple[str, float]]:
    """Return up to k (gene, similarity) pairs among `candidate_genes`
    most similar to `query_gene` by GO-BP Jaccard similarity, sorted
    descending, excluding zero-similarity (no shared annotation) genes."""
    query_terms = gene_to_go.get(query_gene, set())
    if not query_terms:
        return []
    sims = []
    for g in candidate_genes:
        if g == query_gene:
            continue
        sim = jaccard(query_terms, gene_to_go.get(g, set()))
        if sim > 0:
            sims.append((g, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]
