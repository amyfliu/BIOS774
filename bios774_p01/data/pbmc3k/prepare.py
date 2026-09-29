"""
prepare.py — download and standardize the Scanpy PBMC 3k dataset.

Run this LOCALLY (not in a network-restricted sandbox) since it needs to
reach scanpy's backup data URLs (both sc.datasets.pbmc3k() and
sc.datasets.pbmc3k_processed() below).

    cd data/pbmc3k
    python prepare.py

Produces:
    prepared.npz   — X (raw counts, n_cells x n_genes, no preprocessing applied),
                      y (real cell-type annotation, for evaluation only —
                         NOT to be used as a feature or for supervised anything),
                      feature_names (gene symbols), sample_ids (cell barcodes)
    DATA.md        — column/role description for the agent

Deliberately RAW: no normalization, no log transform, no HVG selection, no
scaling. Preprocessing decisions belong to the agent (see AGENTS.md /
.agents/skills/preprocessing/SKILL.md), not this script — this script's only
job is "get the data into a loadable, standardized shape."
"""
import numpy as np
import scanpy as sc


def main():
    adata = sc.datasets.pbmc3k()

    # Real cell-type ground truth: scanpy ships its own official, annotated
    # "answer key" for this exact dataset — the AnnData object at the end of
    # scanpy's PBMC3k clustering tutorial, built from the same raw 10x data
    # and hand-annotated with cell types (not an in-house clustering run by
    # this script). Match purely by cell barcode; that pipeline's own QC step
    # drops a small number of low-quality cells, so not every barcode in
    # `adata` has a match — keep only the ones that do.
    annotated = sc.datasets.pbmc3k_processed()
    labeled_mask = adata.obs_names.isin(annotated.obs_names)
    n_total, n_labeled = adata.n_obs, int(labeled_mask.sum())
    adata = adata[labeled_mask].copy()
    labels = annotated.obs.loc[adata.obs_names, "louvain"].astype(str).to_numpy()

    # Basic non-judgmental cleanup only: drop genes expressed in zero cells
    # among the retained cells (not a preprocessing *decision*, just removing
    # structurally empty columns that would make every DR method choke
    # identically regardless of method).
    sc.pp.filter_genes(adata, min_cells=1)

    X = adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

    np.savez(
        "prepared.npz",
        X=X.astype(np.float32),
        y=labels,
        feature_names=adata.var_names.to_numpy(),
        sample_ids=adata.obs_names.to_numpy(),
    )

    with open("DATA.md", "w") as f:
        f.write(f"""# PBMC 3k — Data Description

Source: 10x Genomics 3k PBMCs, via `scanpy.datasets.pbmc3k()`.

## Contents of prepared.npz
- `X`: ({X.shape[0]}, {X.shape[1]}) raw UMI count matrix (cells x genes).
  NOT normalized, NOT log-transformed, NOT scaled — raw counts.
- `feature_names`: gene symbols, one per column of X.
- `sample_ids`: cell barcodes, one per row of X.
- `y`: real cell-type annotation (e.g. "CD4 T cells", "B cells",
  "CD14+ Monocytes", "NK cells", "CD8 T cells", "FCGR3A+ Monocytes",
  "Dendritic cells", "Megakaryocytes"), sourced from
  scanpy's own official PBMC3k clustering-tutorial dataset
  (`sc.datasets.pbmc3k_processed()`) and matched to these cells by barcode —
  NOT an in-house clustering run by this script. Only the {n_labeled} of the
  original {n_total} cells with a match are kept in prepared.npz; the other
  {n_total - n_labeled} were dropped because that tutorial pipeline's own QC
  step filtered them out and they have no label. Provided ONLY for
  evaluation (e.g. silhouette score) — do not use as a feature, and do not
  let the DR pipeline "cheat" by using it for anything but post-hoc
  evaluation. Treat it as a reasonable, widely-cited community annotation
  for this specific dataset, not an infallible experimentally-validated
  ground truth.

## Known characteristics worth confirming via inspect_data.py, not assuming
- Expected to be sparse (single-cell count data is mostly zeros).
- Expected to be high-dimensional relative to sample count (~genes >> cells
  is common in raw form, though PBMC 3k specifically has ~{X.shape[0]} cells
  and a few thousand genes after the min_cells filter).
- Raw counts are highly skewed (a handful of highly-expressed genes dominate
  variance) — this is exactly the kind of thing a log-transform decision
  should be made about, not assumed.
""")
    print(f"Saved prepared.npz: X {X.shape}, {len(np.unique(labels))} label classes "
          f"({n_total - n_labeled} unlabeled cells dropped)")


if __name__ == "__main__":
    main()
