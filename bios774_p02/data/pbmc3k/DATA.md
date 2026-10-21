# PBMC 3k — Data Description

Source: 10x Genomics 3k PBMCs, via `scanpy.datasets.pbmc3k()`.
Modality: **tabular** (single-cell RNA-seq count matrix).

## Contents of prepared.npz
- `X`: (2638, 16579) raw UMI counts (cells x genes), float32.
  NOT normalized, NOT log-transformed, NOT gene-filtered beyond dropping
  genes detected in zero cells.
- `y`: cell-type annotation from scanpy's official PBMC3k tutorial
  (`sc.datasets.pbmc3k_processed()`, column `louvain`), matched by barcode.
  62 of 2700 cells had no label and were dropped.
  May be used as a conditioning variable when the objective needs
  class-specific generation; it is a community annotation, not a perfect
  ground truth.
- `split`: fixed stratified split (75% train / 25% test,
  seed 774). Fit preprocessing and generative models on `train` ONLY;
  `test` is the untouched real reference for evaluation.
- `feature_names`: gene symbols. `sample_ids`: cell barcodes.

## Class counts
  - CD4 T cells: 1144 total (858 train / 286 test)
  - CD14+ Monocytes: 480 total (360 train / 120 test)
  - B cells: 342 total (256 train / 86 test)
  - CD8 T cells: 316 total (237 train / 79 test)
  - NK cells: 154 total (115 train / 39 test)
  - FCGR3A+ Monocytes: 150 total (113 train / 37 test)
  - Dendritic cells: 37 total (28 train / 9 test)
  - Megakaryocytes: 15 total (11 train / 4 test)

## Things to confirm with inspect_data.py rather than assume
- Very sparse, highly skewed counts; library size varies between cells.
- Genes (p) far outnumber cells (n) — most genes are near-silent.
- Strong class imbalance; the rarest classes have only a handful of test cells,
  so per-class test metrics for them are noisy.
