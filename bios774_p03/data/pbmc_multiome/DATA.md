# PBMC Multiome — paired scRNA-seq + scATAC-seq (10x Genomics)

**Samples:** 10405 single nuclei (human peripheral blood mononuclear cells,
one healthy donor), each measured in **two modalities at once**, so row *i* of
both matrices is the same cell (paired / "same-cell" multimodal data).

| modality | kind | features | stored values | fraction of zero entries |
|---|---|---|---|---|
| `rna` | counts | 25598 genes | raw UMI counts | 0.942 |
| `atac` | counts | 111698 peaks (open-chromatin regions) | raw fragment counts per peak | 0.918 |

Feature tables: `rna_features.tsv` (gene_id, gene_name);
`atac_features.tsv` (peak, chrom, start, end, nearest_genes, peak_type —
nearest genes come from 10x's peak annotation and link peaks to genes for
cross-modal interpretation).

## Labels (for evaluation)

- `label`: coarse immune cell type; `label_fine`: fine cell type.
- Source: CellTypist immune reference models (Immune_All_High → `label`, Immune_All_Low → `label_fine`), majority voting, on log-normalised RNA.
  These are *reference* labels predicted from RNA alone, so they can favour
  RNA-based representations; treat them as a reasonable but imperfect yardstick.
- `label_conf` / `label_fine_conf`: CellTypist's per-cell probability of its label.

| label | train | val | test | total |
|---|---:|---:|---:|---:|
| T cells | 4367 | 936 | 936 | 6239 |
| Monocytes | 1946 | 417 | 417 | 2780 |
| ILC | 410 | 88 | 88 | 586 |
| B cells | 389 | 83 | 83 | 555 |
| DC | 101 | 21 | 21 | 143 |
| pDC | 34 | 7 | 7 | 48 |
| Plasma cells | 27 | 6 | 6 | 39 |
| Megakaryocyte precursor | 11 | 2 | 2 | 15 |

## Splits

Fixed stratified split by `label` (70/15/15, seed 0): fit every
preprocessing step and model on `train` only; use `val` for model choices and
`test` for the final numbers.

## Preparation already applied (not the agent's choice)

- Cell QC: ≥ 200 genes, ≤ 20.0% mitochondrial RNA counts,
  ≥ 1000 ATAC counts in peaks: 10970 → 10405 cells
  (low genes 341, high mito 206,
  low ATAC 101; a cell can fail several).
- Features: genes in ≥ 3 cells (36601 → 25598),
  peaks in ≥ 10 cells (111743 → 111698).
- Nothing else: no normalisation, no feature selection, no TF-IDF.

Other `obs.csv` columns: per-cell QC metrics (`n_counts_rna`, `n_genes_rna`,
`pct_mt`, `n_counts_atac`, `n_peaks_atac`).

Source: https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0 (Cell Ranger ARC 2.0.0).
