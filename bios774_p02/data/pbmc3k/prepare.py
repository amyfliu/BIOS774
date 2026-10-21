"""
prepare.py — download and standardize the Scanpy PBMC 3k dataset (project 2).

Adapted from bios774_p01/data/pbmc3k/prepare.py. Differences for project 2:
  * adds a fixed, stratified train/test `split` column, so every generative
    model is fit on `train` only and evaluated against untouched real `test`
    cells (needed for memorization checks and train-on-X / test-on-real).
  * labels may now be used as a CONDITIONING variable when the objective
    requires class-specific generation (e.g. class balancing).

Run locally (needs internet to reach scanpy's data URLs):

    cd data/pbmc3k && python prepare.py

Produces prepared.npz + DATA.md. Deliberately RAW counts: normalization,
log transform and gene selection are the agent's decisions.
"""
import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

SEED = 774
TEST_FRAC = 0.25


def main():
    adata = sc.datasets.pbmc3k()
    annotated = sc.datasets.pbmc3k_processed()
    labeled_mask = adata.obs_names.isin(annotated.obs_names)
    n_total, n_labeled = adata.n_obs, int(labeled_mask.sum())
    adata = adata[labeled_mask].copy()
    labels = annotated.obs.loc[adata.obs_names, "louvain"].astype(str).to_numpy()
    sc.pp.filter_genes(adata, min_cells=1)

    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=TEST_FRAC, stratify=labels, random_state=SEED)
    split = np.empty(len(labels), dtype="<U5")
    split[tr], split[te] = "train", "test"

    np.savez_compressed(
        "prepared.npz",
        X=X.astype(np.float32),
        y=labels,
        split=split,
        feature_names=adata.var_names.to_numpy(),
        sample_ids=adata.obs_names.to_numpy(),
    )

    names, counts = np.unique(labels, return_counts=True)
    count_lines = "\n".join(
        f"  - {n}: {c} total ({int(((labels == n) & (split == 'train')).sum())} train / "
        f"{int(((labels == n) & (split == 'test')).sum())} test)"
        for n, c in sorted(zip(names, counts), key=lambda t: -t[1])
    )
    with open("DATA.md", "w") as f:
        f.write(f"""# PBMC 3k — Data Description

Source: 10x Genomics 3k PBMCs, via `scanpy.datasets.pbmc3k()`.
Modality: **tabular** (single-cell RNA-seq count matrix).

## Contents of prepared.npz
- `X`: ({X.shape[0]}, {X.shape[1]}) raw UMI counts (cells x genes), float32.
  NOT normalized, NOT log-transformed, NOT gene-filtered beyond dropping
  genes detected in zero cells.
- `y`: cell-type annotation from scanpy's official PBMC3k tutorial
  (`sc.datasets.pbmc3k_processed()`, column `louvain`), matched by barcode.
  {n_total - n_labeled} of {n_total} cells had no label and were dropped.
  May be used as a conditioning variable when the objective needs
  class-specific generation; it is a community annotation, not a perfect
  ground truth.
- `split`: fixed stratified split ({1 - TEST_FRAC:.0%} train / {TEST_FRAC:.0%} test,
  seed {SEED}). Fit preprocessing and generative models on `train` ONLY;
  `test` is the untouched real reference for evaluation.
- `feature_names`: gene symbols. `sample_ids`: cell barcodes.

## Class counts
{count_lines}

## Things to confirm with inspect_data.py rather than assume
- Very sparse, highly skewed counts; library size varies between cells.
- Genes (p) far outnumber cells (n) — most genes are near-silent.
- Strong class imbalance; the rarest classes have only a handful of test cells,
  so per-class test metrics for them are noisy.
""")
    print(f"Saved prepared.npz: X {X.shape}, {len(names)} classes, "
          f"{len(tr)} train / {len(te)} test")


if __name__ == "__main__":
    main()
