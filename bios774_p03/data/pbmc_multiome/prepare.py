"""Prepare the 10x Genomics PBMC Multiome dataset (paired scRNA-seq + scATAC-seq).

    cd data/pbmc_multiome && python prepare.py

Source: "10k Human PBMCs, Multiome v1.0, Chromium X" (Cell Ranger ARC 2.0.0),
https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0

What this script does (and nothing more — every modelling choice is left to
the agent):
  1. downloads the filtered feature-barcode matrix (.h5, both modalities)
     and the ATAC peak annotation into raw/;
  2. splits the matrix into RNA (gene counts) and ATAC (peak counts);
  3. light cell QC (thresholds below, recorded in DATA.md) and removal of
     features seen in almost no cells;
  4. reference cell-type labels with CellTypist (immune reference models) on
     the RNA counts — used for evaluation, never as ground truth biology;
  5. a fixed stratified train/val/test split (70/15/15, seed 0);
  6. writes rna.npz, atac.npz, feature tables, obs.csv, dataset.json, DATA.md.

Options: --h5 / --annotation use already-downloaded files; --labels-csv gives
your own labels (columns: barcode,label,label_fine) instead of CellTypist;
--min-genes etc. change the QC thresholds (defaults are for the real data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dataset_io import download, label_table, stratified_split, write_manifest  # noqa: E402

BASE = ("https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
        "10k_PBMC_Multiome_nextgem_Chromium_X/10k_PBMC_Multiome_nextgem_Chromium_X_")
H5_URL = BASE + "filtered_feature_bc_matrix.h5"
ANNOT_URL = BASE + "atac_peak_annotation.tsv"
PAGE = "https://www.10xgenomics.com/datasets/10-k-human-pbm-cs-multiome-v-1-0-chromium-x-1-standard-2-0-0"

# QC thresholds (light; the filtered matrix is already cell-called by Cell Ranger ARC)
MIN_GENES = 200          # genes detected per cell
MAX_PCT_MT = 20.0        # % RNA counts from mitochondrial genes
MIN_ATAC_COUNTS = 1000   # ATAC fragments in peaks per cell
MIN_CELLS_GENE = 3       # keep genes detected in >= 3 cells
MIN_CELLS_PEAK = 10      # keep peaks detected in >= 10 cells
SEED = 0


def read_10x(h5: Path):
    import scanpy as sc

    ad = sc.read_10x_h5(str(h5), gex_only=False)
    ad.var_names_make_unique()
    ft = ad.var["feature_types"].astype(str)
    rna = ad[:, (ft == "Gene Expression").values].copy()
    atac = ad[:, (ft == "Peaks").values].copy()
    if rna.n_vars == 0 or atac.n_vars == 0:
        raise SystemExit(f"Expected 'Gene Expression' and 'Peaks' features, found {sorted(ft.unique())}")
    return rna, atac


def qc(rna, atac, MIN_GENES=MIN_GENES, MAX_PCT_MT=MAX_PCT_MT, MIN_ATAC_COUNTS=MIN_ATAC_COUNTS,
       MIN_CELLS_GENE=MIN_CELLS_GENE, MIN_CELLS_PEAK=MIN_CELLS_PEAK):
    Xr = sp.csr_matrix(rna.X)
    Xa = sp.csr_matrix(atac.X)
    names = rna.var_names.str.upper()
    mt = np.asarray(names.str.startswith("MT-"))
    n_counts_rna = np.asarray(Xr.sum(1)).ravel()
    n_genes = np.asarray((Xr > 0).sum(1)).ravel()
    pct_mt = 100 * np.asarray(Xr[:, mt].sum(1)).ravel() / np.maximum(n_counts_rna, 1)
    n_counts_atac = np.asarray(Xa.sum(1)).ravel()
    n_peaks = np.asarray((Xa > 0).sum(1)).ravel()

    keep = (n_genes >= MIN_GENES) & (pct_mt <= MAX_PCT_MT) & (n_counts_atac >= MIN_ATAC_COUNTS)
    stats = {
        "cells_before_qc": int(len(keep)),
        "removed_low_genes": int((n_genes < MIN_GENES).sum()),
        "removed_high_mt": int((pct_mt > MAX_PCT_MT).sum()),
        "removed_low_atac": int((n_counts_atac < MIN_ATAC_COUNTS).sum()),
        "cells_after_qc": int(keep.sum()),
    }
    obs = pd.DataFrame({
        "sample_id": rna.obs_names.values,
        "n_counts_rna": n_counts_rna, "n_genes_rna": n_genes, "pct_mt": np.round(pct_mt, 2),
        "n_counts_atac": n_counts_atac, "n_peaks_atac": n_peaks,
    })[keep].reset_index(drop=True)
    if keep.sum() < 50:
        raise SystemExit(f"QC kept only {int(keep.sum())} cells: {stats}. Check the thresholds (see --help).")
    Xr, Xa = Xr[keep], Xa[keep]

    gkeep = np.asarray((Xr > 0).sum(0)).ravel() >= MIN_CELLS_GENE
    pkeep = np.asarray((Xa > 0).sum(0)).ravel() >= MIN_CELLS_PEAK
    stats.update(genes_before=int(Xr.shape[1]), genes_after=int(gkeep.sum()),
                 peaks_before=int(Xa.shape[1]), peaks_after=int(pkeep.sum()))
    rvar = rna.var.iloc[np.flatnonzero(gkeep)]
    avar = atac.var.iloc[np.flatnonzero(pkeep)]
    return obs, Xr[:, gkeep], Xa[:, pkeep], rvar, avar, stats


def celltypist_labels(Xr, genes, barcodes):
    """Reference labels from CellTypist immune models (majority voting)."""
    try:
        import anndata as an
        import celltypist
        import scanpy as sc
        from celltypist import models
    except ImportError:
        raise SystemExit("CellTypist is needed for reference labels: pip install celltypist")

    ad = an.AnnData(X=Xr.astype(np.float32).copy(), obs=pd.DataFrame(index=barcodes),
                    var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    out = {}
    for level, model in (("label", "Immune_All_High.pkl"), ("label_fine", "Immune_All_Low.pkl")):
        print(f"  CellTypist {model} (majority voting) ...")
        models.download_models(model=model, force_update=False)
        pred = celltypist.annotate(ad, model=model, majority_voting=True)
        lab = pred.predicted_labels
        out[level] = lab["majority_voting"].astype(str).values
        out[level + "_conf"] = np.round(pred.probability_matrix.max(axis=1).values, 3)
    return pd.DataFrame(out, index=barcodes)


def peak_table(avar, annot_path: Path | None) -> pd.DataFrame:
    peaks = avar.index.astype(str)
    parts = peaks.str.replace("-", ":", n=1).str.split(":")
    tab = pd.DataFrame({
        "peak": peaks,
        "chrom": [p[0] for p in parts],
        "start": [int(p[1]) for p in parts],
        "end": [int(p[2]) for p in parts],
    })
    tab["nearest_genes"] = ""
    tab["peak_type"] = ""
    if annot_path is not None and Path(annot_path).exists():
        ann = pd.read_csv(annot_path, sep="\t", comment="#", header=None)
        cols = ["chrom", "start", "end", "gene", "distance", "peak_type"]
        ann = ann.iloc[:, :6]
        ann.columns = cols[: ann.shape[1]]
        if str(ann.iloc[0, 1]).lower() == "start":  # header row present
            ann = ann.iloc[1:]
        ann["peak"] = ann["chrom"].astype(str) + ":" + ann["start"].astype(str) + "-" + ann["end"].astype(str)
        g = ann.groupby("peak").agg(
            nearest_genes=("gene", lambda s: ";".join(sorted({str(x) for x in s if str(x) != "nan"}))),
            peak_type=("peak_type", lambda s: ";".join(sorted({str(x) for x in s if str(x) != "nan"}))))
        tab = tab.drop(columns=["nearest_genes", "peak_type"]).merge(g, left_on="peak", right_index=True, how="left")
        tab[["nearest_genes", "peak_type"]] = tab[["nearest_genes", "peak_type"]].fillna("")
    return tab


def write_data_md(folder: Path, obs: pd.DataFrame, stats: dict, n_genes: int, n_peaks: int,
                  sparsity: dict, label_source: str, th: dict) -> None:
    MIN_GENES, MAX_PCT_MT, MIN_ATAC_COUNTS = th["min_genes"], th["max_pct_mt"], th["min_atac_counts"]
    MIN_CELLS_GENE, MIN_CELLS_PEAK = th["min_cells_gene"], th["min_cells_peak"]
    rna_note = ("" if "CellTypist" not in label_source else
                "\n  These are *reference* labels predicted from RNA alone, so they can favour\n"
                "  RNA-based representations; treat them as a reasonable but imperfect yardstick.\n"
                "- `label_conf` / `label_fine_conf`: CellTypist's per-cell probability of its label.")
    md = f"""# PBMC Multiome — paired scRNA-seq + scATAC-seq (10x Genomics)

**Samples:** {len(obs)} single nuclei (human peripheral blood mononuclear cells,
one healthy donor), each measured in **two modalities at once**, so row *i* of
both matrices is the same cell (paired / "same-cell" multimodal data).

| modality | kind | features | stored values | fraction of zero entries |
|---|---|---|---|---|
| `rna` | counts | {n_genes} genes | raw UMI counts | {sparsity['rna']:.3f} |
| `atac` | counts | {n_peaks} peaks (open-chromatin regions) | raw fragment counts per peak | {sparsity['atac']:.3f} |

Feature tables: `rna_features.tsv` (gene_id, gene_name);
`atac_features.tsv` (peak, chrom, start, end, nearest_genes, peak_type —
nearest genes come from 10x's peak annotation and link peaks to genes for
cross-modal interpretation).

## Labels (for evaluation)

- `label`: coarse immune cell type; `label_fine`: fine cell type.
- Source: {label_source}.{rna_note}

{label_table(obs, 'label')}

## Splits

Fixed stratified split by `label` (70/15/15, seed {SEED}): fit every
preprocessing step and model on `train` only; use `val` for model choices and
`test` for the final numbers.

## Preparation already applied (not the agent's choice)

- Cell QC: ≥ {MIN_GENES} genes, ≤ {MAX_PCT_MT}% mitochondrial RNA counts,
  ≥ {MIN_ATAC_COUNTS} ATAC counts in peaks: {stats['cells_before_qc']} → {stats['cells_after_qc']} cells
  (low genes {stats['removed_low_genes']}, high mito {stats['removed_high_mt']},
  low ATAC {stats['removed_low_atac']}; a cell can fail several).
- Features: genes in ≥ {MIN_CELLS_GENE} cells ({stats['genes_before']} → {stats['genes_after']}),
  peaks in ≥ {MIN_CELLS_PEAK} cells ({stats['peaks_before']} → {stats['peaks_after']}).
- Nothing else: no normalisation, no feature selection, no TF-IDF.

Other `obs.csv` columns: per-cell QC metrics (`n_counts_rna`, `n_genes_rna`,
`pct_mt`, `n_counts_atac`, `n_peaks_atac`).

Source: {PAGE} (Cell Ranger ARC 2.0.0).
"""
    (folder / "DATA.md").write_text(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", type=Path, help="already-downloaded filtered_feature_bc_matrix.h5")
    ap.add_argument("--annotation", type=Path, help="already-downloaded atac_peak_annotation.tsv")
    ap.add_argument("--labels-csv", type=Path, help="your own labels: barcode,label,label_fine")
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--min-genes", type=int, default=MIN_GENES)
    ap.add_argument("--max-pct-mt", type=float, default=MAX_PCT_MT)
    ap.add_argument("--min-atac-counts", type=int, default=MIN_ATAC_COUNTS)
    ap.add_argument("--min-cells-gene", type=int, default=MIN_CELLS_GENE)
    ap.add_argument("--min-cells-peak", type=int, default=MIN_CELLS_PEAK)
    args = ap.parse_args()
    th = {"min_genes": args.min_genes, "max_pct_mt": args.max_pct_mt, "min_atac_counts": args.min_atac_counts,
          "min_cells_gene": args.min_cells_gene, "min_cells_peak": args.min_cells_peak}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("1/5 download")
    h5 = args.h5 or download(H5_URL, out / "raw" / "filtered_feature_bc_matrix.h5")
    annot = args.annotation
    if annot is None:
        try:
            annot = download(ANNOT_URL, out / "raw" / "atac_peak_annotation.tsv")
        except SystemExit as e:  # annotation is optional
            print(f"  (peak annotation unavailable, continuing without nearest genes)\n  {e}")
            annot = None

    print("2/5 read + split modalities")
    rna, atac = read_10x(h5)
    print(f"  {rna.n_obs} cells, {rna.n_vars} genes, {atac.n_vars} peaks")

    print("3/5 QC")
    obs, Xr, Xa, rvar, avar, stats = qc(rna, atac, th["min_genes"], th["max_pct_mt"], th["min_atac_counts"],
                                        th["min_cells_gene"], th["min_cells_peak"])
    stats["thresholds"] = th
    print("  " + json.dumps(stats))

    print("4/5 reference labels")
    if args.labels_csv:
        lab = pd.read_csv(args.labels_csv).set_index("barcode")
        lab = lab.reindex(obs["sample_id"])
        if lab["label"].isna().any():
            raise SystemExit(f"--labels-csv is missing {int(lab['label'].isna().sum())} barcodes")
        if "label_fine" not in lab:
            lab["label_fine"] = lab["label"]
        label_source = f"user-provided file `{args.labels_csv.name}`"
    else:
        gene_names = rvar.index.astype(str)
        lab = celltypist_labels(Xr, gene_names, obs["sample_id"].values)
        label_source = ("CellTypist immune reference models (Immune_All_High → `label`, "
                        "Immune_All_Low → `label_fine`), majority voting, on log-normalised RNA")
    for c in lab.columns:
        obs[c] = lab[c].values
    obs["split"] = stratified_split(obs["label"], seed=SEED)

    print("5/5 write")
    sp.save_npz(out / "rna.npz", Xr.astype(np.int32).tocsr(), compressed=True)
    sp.save_npz(out / "atac.npz", Xa.astype(np.int32).tocsr(), compressed=True)
    gene_ids = rvar["gene_ids"].astype(str).values if "gene_ids" in rvar else rvar.index.astype(str)
    pd.DataFrame({"gene_id": gene_ids, "gene_name": rvar.index.astype(str)}).to_csv(
        out / "rna_features.tsv", sep="\t", index=False)
    peak_table(avar, annot).to_csv(out / "atac_features.tsv", sep="\t", index=False)

    cols = ["sample_id", "split", "label", "label_fine"] + [c for c in obs.columns if c not in
                                                            ("sample_id", "split", "label", "label_fine")]
    obs = obs[cols]
    manifest = {
        "name": "pbmc_multiome",
        "description": "10x PBMC Multiome: paired scRNA-seq gene counts + scATAC-seq peak counts per nucleus",
        "pairing": "same cell measured in both modalities (row i of each matrix = obs row i)",
        "obs": "obs.csv", "id_column": "sample_id", "split_column": "split",
        "label_columns": [c for c in ("label", "label_fine") if c in obs],
        "primary_label": "label",
        "modalities": [
            {"name": "rna", "kind": "counts", "file": "rna.npz", "features": "rna_features.tsv",
             "description": "gene expression, raw UMI counts"},
            {"name": "atac", "kind": "counts", "file": "atac.npz", "features": "atac_features.tsv",
             "description": "chromatin accessibility, raw counts per peak"},
        ],
        "qc": stats,
        "source": {"url": PAGE, "h5": H5_URL, "license": "10x Genomics public dataset (CC BY 4.0)",
                   "labels": label_source},
    }
    sparsity = {"rna": 1 - Xr.nnz / (Xr.shape[0] * Xr.shape[1]),
                "atac": 1 - Xa.nnz / (Xa.shape[0] * Xa.shape[1])}
    manifest["sparsity"] = {k: round(v, 4) for k, v in sparsity.items()}
    write_manifest(out, manifest, obs)
    write_data_md(out, obs, stats, Xr.shape[1], Xa.shape[1], sparsity, label_source, th)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
