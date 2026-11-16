# Dataset Card

Verified from primary sources on 2026-11-05. Nothing below is assumed —
where a fact could not be confirmed from a primary source, it is marked
**unverified** rather than guessed.

## 1. Arc Institute Virtual Cell Atlas / Virtual Cell Challenge

- **Repo:** https://github.com/ArcInstitute/arc-virtual-cell-atlas
- **Website:** https://arcinstitute.org/tools/virtualcellatlas
- **Challenge used for this project:** "Virtual Cell Challenge" 2025
  (README describes it as Arc's inaugural challenge). No 2026 challenge
  found in the repo as of this check.

**License:** The GitHub repo itself contains no LICENSE file or license
statement. The arcinstitute.org tools page states, for Tahoe-100M,
scBaseCount, and the Virtual Cell Challenge 2025 dataset: "© 2025. This
work is openly licensed via CC0 1.0" (linking to the CC0 1.0 legal code).
This is a website statement, not a repo-embedded license — treat it as the
operative license but note the lack of a repo-level LICENSE file.

**Access:** Data lives in `gs://arc-institute-virtual-cell-atlas`, stated in
the README as "publicly readable and doesn't require a Google Cloud
project or billing account." Accessible via `gcloud storage` or `gcsfs`
with `storage_options={"token": "anon"}`. Requester-pays status and
free-tier limits for this specific bucket are **unverified** (the README
only asks users who see unexpected charges to open an issue). Prefix
sizes reported in the README: `scbasecount/` 39.5 TiB, `tahoe100M/` 316
GiB, `virtual-cell-challenge/` 5.3 TiB (5.28 TiB of that is raw FASTQ).
Total atlas size: "over 602 million cells."

**Format:** Count matrices as h5ad (AnnData); metadata as Parquet & CSV.

**Challenge dataset specifics (H1 hESC):** ~300,000 cells, 300 target
genes (per README). Data is organized by release year and split into
`2025/train`, `2025/validation`, `2025/test`. Per-split cell/target counts
and the contents of the test set are **unverified** — not stated in the
README as fetched. Training folder contains `adata_Training.h5ad` and
`pert_counts_Training.csv`.

**Official splits:** Yes — Arc provides train/validation/test folders for
the 2025 challenge. If used, this project would replicate Arc's official
split separately from our own `held_out_target_split`, per the plan's
instruction not to conflate the two.

**Verified `adata_Training.h5ad` schema** (inspected 2026-11-05 via
`gcsfs`/`h5py` byte-range reads against the live anonymous bucket, without
downloading the full 14.42 GiB file):

- Shape: 221,273 cells × 18,080 genes. `X` is a CSR sparse matrix
  (`float32` data, ~1.93e9 stored nonzeros).
- `obs` columns: `target_gene` (categorical, 151 categories, including
  `non-targeting` as the control label — there is **no separate
  `is_control` boolean column**), `guide_id` (categorical, e.g.
  `ACAT2_P1P2_A|ACAT2_P1P2_B`), `batch` (categorical, e.g. `Flex_1_01`).
  No `cell_type` or `donor_id` column is present in this file.
- `var` columns: `gene_id` only (indexed by `_index`).
- `pert_counts_Training.csv` (2.8 KiB, downloaded and inspected directly):
  columns `target_gene,n_cells,median_umi_per_cell`, e.g.
  `TMSB4X,4760,53652.0`. 151 target genes total, matching the h5ad
  `target_gene` category count.

**Mapping to our canonical schema** (`virtualcell.data.adapters`):
`perturbation_id = target_gene` (with `non-targeting` renamed to
`control` to match our schema's convention), `is_control = (target_gene
== 'non-targeting')`, `batch_id = batch`. `cell_type` and `donor_id` are
**not available** in this file and must be left unset — any config or
code claiming otherwise would be fabricating metadata that does not
exist in the source.

## 2. scPerturb

- **Website:** https://www.sanderlab.org/scPerturb/ and
  https://projects.sanderlab.org/scperturb
- **Code repo:** https://github.com/sanderlab/scPerturb
- **Paper:** Peidli et al., *Nature Methods* 2024, vol. 21, pp. 531–540,
  doi 10.1038/s41592-023-02144-y.

**License:**
- Code (GitHub repo, excluding the `pertpy`/R package folders which have
  their own licenses): MIT, "Copyright (c) 2022 Stefan Peidli."
- Data (Zenodo): the current RNA record,
  [zenodo.org/records/13350497](https://zenodo.org/records/13350497)
  (v1.4), lists its license as **Creative Commons Attribution 4.0
  International (CC BY 4.0)**. The older v1.0 record
  (zenodo.org/records/7041849) also lists CC BY 4.0. No separate license
  text appears on the scPerturb website or GitHub README itself — this
  came directly from the Zenodo records.
- CC BY 4.0 requires attribution to Peidli et al. 2024 if used; the
  underlying original studies the data was harmonized from may carry
  additional terms not checked here.

**Format:** ~54 `.h5ad` files (gzip-compressed via scanpy 1.9.1), ~43.0 GB
total in the current record; 26.9 GB in the older v1.0 record.

**Perturbation/control labeling:** Not confirmed via a schema document —
the dataset_info explorer page returned HTTP 404 at the URL checked. The
only evidence found is a usage example in the GitHub README using
`obs_key='perturbation'` and a literal value `'control'` for control
cells. This is illustrative, not a verified schema spec, and **must be
checked directly against a downloaded `.h5ad` file's `obs` columns**
before building an adapter — do not assume these column/value names
generalize across all 44 harmonized datasets without checking the
specific file in use.

**Official splits:** None found.

## Decision for this project

**Current status (Milestones 10–35):** The project uses the **Replogle et al.
2022 K562 essential gene screen** as its primary dataset
(DOI: 10.25452/figshare.plus.20029387). Both single-cell and bulk h5ad files
have been downloaded and directly inspected. Key schema facts confirmed from
the actual files:

- `adata.obs["gene"]` holds gene names; control cells have value
  `"non-targeting"`.
- The bulk file (`K562_essential_raw_bulk_01.h5ad`) has `obs["gene"]` with one
  row per gene mean signature.
- v10+ scripts use `adata.obs["gene"]` directly; there is no `perturbation_id`
  column in these files (the legacy scaffold used that name).

**Reproduction requirements for v25** (best result, 0.3557 Δ Pearson):

```
data/replogle/K562_essential_raw_singlecell_01.h5ad   (~9 GB)
data/replogle_k562_essential/K562_essential_raw_bulk_01.h5ad  (~500 MB)
data/esm2/esm2_embeddings.npz
data/gene2vec/gene2vec_dim_200_iter_9.txt
data/string/9606.protein.links.v12.0.txt.gz
data/string/9606.protein.info.v12.0.txt.gz
outputs/replogle_genelevel_v10/mean_delta_cache.npz   (generated by v10)
outputs/replogle_genelevel_v10/mean_delta_cache_genes.json  (generated by v10)
```

Earlier notes below about schema uncertainty predate direct inspection and
apply only to the Arc/scPerturb datasets.
