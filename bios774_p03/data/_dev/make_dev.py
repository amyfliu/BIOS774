"""Build tiny synthetic stand-ins for both datasets, for testing without downloads.

    python data/_dev/make_dev.py

It fakes the *raw* downloads (a 10x Multiome .h5 + peak annotation; the ROCOv2
Zenodo CSVs + image zips) and then runs the real prepare.py scripts on them, so
the preparation code itself is exercised. Results:

    data/_dev/pbmc_multiome/   ~600 cells, 5 cell types, RNA + ATAC with shared signal
    data/_dev/rocov2/          ~420 images, 4 modalities, captions that mention them

The signal is simple on purpose (each cell type / imaging modality has its own
marker features or visual pattern), so tools can be checked for "learns
something" vs "learns nothing". Never use these for reported results.
"""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
rng = np.random.default_rng(774)


# --------------------------------------------------------------------------- #
# fake 10x Multiome
# --------------------------------------------------------------------------- #
def write_10x_h5(path: Path, X, barcodes, ids, names, ftypes, intervals):
    import h5py
    import scipy.sparse as sp

    Xc = sp.csc_matrix(X.T)  # 10x stores features x barcodes, CSC by barcode
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        g.create_dataset("data", data=Xc.data.astype(np.int32))
        g.create_dataset("indices", data=Xc.indices.astype(np.int64))
        g.create_dataset("indptr", data=Xc.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(Xc.shape, dtype=np.int32))
        ft = g.create_group("features")
        ft.create_dataset("_all_tag_keys", data=np.array(["genome", "interval"], dtype="S"))
        ft.create_dataset("id", data=np.array(ids, dtype="S"))
        ft.create_dataset("name", data=np.array(names, dtype="S"))
        ft.create_dataset("feature_type", data=np.array(ftypes, dtype="S"))
        ft.create_dataset("genome", data=np.array(["GRCh38"] * len(ids), dtype="S"))
        ft.create_dataset("interval", data=np.array(intervals, dtype="S"))


def fake_pbmc(raw: Path) -> tuple[Path, Path, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    types = ["CD4 T", "CD8 T", "B", "NK", "Monocyte"]
    n_per = [180, 110, 120, 80, 130]
    labels = np.repeat(types, n_per)
    n = len(labels)
    k = len(types)
    n_genes, n_mt, n_peaks = 300, 13, 1500
    # each type: 20 marker genes; each marker gene has 3 nearby peaks that open with it
    gene_type = np.full(n_genes, -1)
    for t in range(k):
        gene_type[t * 20:(t + 1) * 20] = t
    lib = rng.lognormal(7.3, 0.4, n)
    base = rng.gamma(0.6, 1.0, n_genes)
    rates = np.tile(base, (n, 1))
    for i, lab in enumerate(labels):
        t = types.index(lab)
        rates[i, gene_type == t] *= 8
    rates[:, -n_mt:] *= 3
    rates = rates / rates.sum(1, keepdims=True) * lib[:, None]
    rna = rng.poisson(rates)
    rna[:5, :] = 0  # 5 empty-ish cells that QC must remove
    rna[:5, :50] = 1

    peak_gene = np.full(n_peaks, -1)
    peak_gene[: k * 20 * 3] = np.repeat(np.arange(k * 20), 3)
    acc = np.full((n, n_peaks), 0.02)
    expr_type = np.array([types.index(x) for x in labels])
    for p in range(k * 20 * 3):
        acc[expr_type == gene_type[peak_gene[p]], p] = 0.35
    acc[:, k * 20 * 3:] = rng.uniform(0.005, 0.08, n_peaks - k * 20 * 3)
    depth = rng.lognormal(0, 0.3, n)[:, None]
    atac = rng.binomial(2, np.clip(acc * depth, 0, 1))
    atac[:, -40:] = 0  # never-open peaks that feature filtering must remove
    atac[np.arange(n) % 97 == 3, :] = 0  # a few cells with too little ATAC

    gene_names = [f"GENE{j}" for j in range(n_genes - n_mt)] + [f"MT-ND{j}" for j in range(n_mt)]
    gene_names[0] = "CD3E"  # duplicate-safe, realistic-looking names
    gene_ids = [f"ENSG{j:011d}" for j in range(n_genes)]
    starts = np.sort(rng.choice(np.arange(10_000, 50_000_000, 1000), n_peaks, replace=False))
    peaks = [f"chr{1 + j % 22}:{s}-{s + 500}" for j, s in enumerate(starts)]
    intervals = [f"chr1:{1000 * j}-{1000 * j + 900}" for j in range(n_genes)] + peaks
    X = np.hstack([rna, atac])
    barcodes = [f"AAAC{j:08d}-1" for j in range(n)]
    h5 = raw / "filtered_feature_bc_matrix.h5"
    write_10x_h5(h5, X, barcodes, gene_ids + peaks, gene_names + peaks,
                 ["Gene Expression"] * n_genes + ["Peaks"] * n_peaks, intervals)

    rows = []
    for p, name in enumerate(peaks):
        c, se = name.split(":")
        s, e = se.split("-")
        g = gene_names[peak_gene[p]] if peak_gene[p] >= 0 else gene_names[p % (n_genes - n_mt)]
        rows.append((c, s, e, g, 0 if peak_gene[p] >= 0 else 5000, "promoter" if peak_gene[p] >= 0 else "distal"))
    annot = raw / "atac_peak_annotation.tsv"
    pd.DataFrame(rows, columns=["chrom", "start", "end", "gene", "distance", "peak_type"]).to_csv(
        annot, sep="\t", index=False)
    labs = raw / "labels.csv"
    pd.DataFrame({"barcode": barcodes, "label": labels,
                  "label_fine": [f"{x} {'naive' if i % 2 else 'memory'}" if "T" in x else x
                                 for i, x in enumerate(labels)]}).to_csv(labs, index=False)
    return h5, annot, labs


# --------------------------------------------------------------------------- #
# fake ROCOv2
# --------------------------------------------------------------------------- #
MODS = {  # CUI, canonical name, short, caption phrase
    "C0040405": ("X-Ray Computed Tomography", "CT", "axial CT scan"),
    "C0024485": ("Magnetic Resonance Imaging", "MRI", "T2-weighted MRI"),
    "C1306645": ("Plain x-ray", "X-ray", "chest radiograph"),
    "C0041618": ("Ultrasonography", "Ultrasound", "ultrasound image"),
}
ANAT = {"C0817096": "Chest", "C0000726": "Abdomen", "C0018670": "Head", "C0024091": "Lumbar spine"}
FINDINGS = ["showing a mass", "with a large lesion (arrow)", "demonstrating fluid collection",
            "without abnormal findings", "revealing a fracture", "showing an enhancing nodule"]


def fake_image(short: str, size=(320, 288)):
    from PIL import Image

    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    if short == "CT":
        img = 40 + 160 * (((xx - w / 2) ** 2 / (0.4 * w) ** 2 + (yy - h / 2) ** 2 / (0.4 * h) ** 2) < 1)
    elif short == "MRI":
        img = 90 + 60 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
    elif short == "X-ray":
        img = 30 + 180 * (xx / w) * (1 - np.abs(yy - h / 2) / h)
    else:  # ultrasound: fan-shaped speckle
        ang = np.abs(np.arctan2(xx - w / 2, yy + 1)) < 0.6
        img = ang * rng.gamma(2.0, 40.0, (h, w))
    img = np.clip(img + rng.normal(0, 12, (h, w)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def fake_rocov2(raw: Path, n_split=None) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    n_split = n_split or {"train": 300, "valid": 60, "test": 60}
    names = [{"CUI": c, "Canonical name": v[0]} for c, v in MODS.items()] + \
            [{"CUI": c, "Canonical name": v} for c, v in ANAT.items()]
    pd.DataFrame(names).to_csv(raw / "cui_mapping.csv", index=False)
    mod_keys, anat_keys = list(MODS), list(ANAT)
    for split, n in n_split.items():
        caps, cons, mans = [], [], []
        with zipfile.ZipFile(raw / f"{split}_images.zip", "w") as z:
            for j in range(n):
                iid = f"ROCOv2_2023_{split}_{j:06d}"
                m = mod_keys[j % len(mod_keys)] if j % 23 else mod_keys[(j + 1) % 4]
                a = anat_keys[rng.integers(len(anat_keys))]
                caps.append((iid, f"{MODS[m][2].capitalize()} of the {ANAT[a].lower()} "
                                  f"{FINDINGS[rng.integers(len(FINDINGS))]}."))
                cons.append((iid, f"{m};{a}"))
                man = f"{m};{a}"
                if j == 7:
                    man = a  # no modality concept -> must be dropped
                if j == 11:
                    man = f"{m};{mod_keys[(mod_keys.index(m) + 1) % 4]}"  # two modalities -> dropped
                mans.append((iid, man))
                z.writestr(f"{split}/{iid}.jpg", fake_image(MODS[m][1]))
        pd.DataFrame(caps, columns=["ID", "Caption"]).to_csv(raw / f"{split}_captions.csv", index=False)
        pd.DataFrame(cons, columns=["ID", "CUIs"]).to_csv(raw / f"{split}_concepts.csv", index=False)
        pd.DataFrame(mans, columns=["ID", "CUIs"]).to_csv(raw / f"{split}_concepts_manual.csv", index=False)
    return raw


def main():
    py = sys.executable
    h5, annot, labs = fake_pbmc(HERE / "raw_pbmc")
    subprocess.run([py, str(DATA / "pbmc_multiome" / "prepare.py"), "--h5", str(h5), "--annotation", str(annot),
                    "--labels-csv", str(labs), "--out", str(HERE / "pbmc_multiome"),
                    "--min-genes", "100", "--min-atac-counts", "30", "--min-cells-peak", "3"], check=True)
    raw = fake_rocov2(HERE / "raw_rocov2")
    subprocess.run([py, str(DATA / "rocov2" / "prepare.py"), "--raw-dir", str(raw),
                    "--max-train", "0", "--max-val", "0", "--max-test", "0",
                    "--out", str(HERE / "rocov2")], check=True)
    print("dev datasets ready: data/_dev/pbmc_multiome, data/_dev/rocov2")


if __name__ == "__main__":
    main()
