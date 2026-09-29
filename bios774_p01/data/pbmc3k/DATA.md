# PBMC 3k — Data Description

Source: 10x Genomics 3k PBMCs, via `scanpy.datasets.pbmc3k()`.

## Contents of prepared.npz
- `X`: (2638, 16579) raw UMI count matrix (cells x genes).
  NOT normalized, NOT log-transformed, NOT scaled — raw counts.
- `feature_names`: gene symbols, one per column of X.
- `sample_ids`: cell barcodes, one per row of X.
- `y`: real cell-type annotation (e.g. "CD4 T cells", "B cells",
  "CD14+ Monocytes", "NK cells", "CD8 T cells", "FCGR3A+ Monocytes",
  "Dendritic cells", "Megakaryocytes"), sourced from
  scanpy's own official PBMC3k clustering-tutorial dataset
  (`sc.datasets.pbmc3k_processed()`) and matched to these cells by barcode —
  NOT an in-house clustering run by this script. Only the 2638 of the
  original 2700 cells with a match are kept in prepared.npz; the other
  62 were dropped because that tutorial pipeline's own QC
  step filtered them out and they have no label. Provided ONLY for
  evaluation (e.g. silhouette score) — do not use as a feature, and do not
  let the DR pipeline "cheat" by using it for anything but post-hoc
  evaluation. Treat it as a reasonable, widely-cited community annotation
  for this specific dataset, not an infallible experimentally-validated
  ground truth.

## Known characteristics worth confirming via inspect_data.py, not assuming
- Expected to be sparse (single-cell count data is mostly zeros).
- Expected to be high-dimensional relative to sample count (~genes >> cells
  is common in raw form, though PBMC 3k specifically has ~2638 cells
  and a few thousand genes after the min_cells filter).
- Raw counts are highly skewed (a handful of highly-expressed genes dominate
  variance) — this is exactly the kind of thing a log-transform decision
  should be made about, not assumed.
