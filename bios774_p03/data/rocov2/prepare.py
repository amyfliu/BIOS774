"""Prepare ROCOv2 (Radiology Objects in COntext v2): radiology images + captions.

    cd data/rocov2 && python prepare.py            # ~6.4 GB download, once

Source: Zenodo record 10821435 (CC BY-NC 4.0), https://zenodo.org/records/10821435
Rückert et al. (2024), Scientific Data 11:688.

What this script does (and nothing more — every modelling choice is left to
the agent):
  1. downloads captions, UMLS concepts, manually curated concepts, the CUI
     name table and the three image archives into raw/;
  2. derives each image's imaging-modality label (CT, MRI, X-ray, ...) from
     the radiologist-checked manual concepts;
  3. keeps ROCOv2's own train / valid / test split (valid -> "val") and draws
     a fixed random subset of each (default 20,000 / 2,000 / 2,000; seed 0)
     so training fits on a laptop;
  4. copies the selected images out of the archives, resized so the shorter
     side is 256 px (aspect ratio kept), to images/<split>/<ID>.jpg;
  5. writes obs.csv (image path, caption, labels, all concepts),
     dataset.json and DATA.md.

Options: --raw-dir points at a folder that already holds the Zenodo files;
--max-train/--max-val/--max-test change the subset sizes (0 = all).
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dataset_io import df_to_md, download, label_table, write_manifest  # noqa: E402

RECORD = "https://zenodo.org/records/10821435"
FILES = ["cui_mapping.csv"] + [f"{s}_{k}" for s in ("train", "valid", "test")
                               for k in ("captions.csv", "concepts.csv", "concepts_manual.csv", "images.zip")]
SPLIT_NAME = {"train": "train", "valid": "val", "test": "test"}
SHORT_SIDE = 256
SEED = 0

# Ordered: the first pattern found in a concept's canonical name wins
# ("X-Ray Computed Tomography" must become CT, not X-ray).
MODALITY_PATTERNS = [
    ("computed tomography", "CT"),
    ("magnetic resonance", "MRI"),
    ("positron", "PET"),
    ("single-photon", "SPECT"),
    ("ultrason", "Ultrasound"),
    ("angiogra", "Angiography"),
    ("mammogra", "Mammography"),
    ("fluoroscop", "Fluoroscopy"),
    ("scintigra", "Nuclear medicine"),
    ("radionuclide", "Nuclear medicine"),
    ("x-ray", "X-ray"),
    ("radiograph", "X-ray"),
]


def _cols(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """ROCOv2 CSVs: first column = image ID, second = value. Normalise names."""
    df = df.iloc[:, :n].copy()
    df.columns = ["ID", "value"][:n]
    df["ID"] = df["ID"].astype(str).str.strip()
    return df


def modality_of(name: str) -> str | None:
    low = str(name).lower()
    for pat, short in MODALITY_PATTERNS:
        if pat in low:
            return short
    return None


def load_split(raw: Path, split: str, cui_name: dict, mod_cuis: dict) -> pd.DataFrame:
    cap = _cols(pd.read_csv(raw / f"{split}_captions.csv"), 2).rename(columns={"value": "caption"})
    con = _cols(pd.read_csv(raw / f"{split}_concepts.csv"), 2).rename(columns={"value": "cuis"})
    man = _cols(pd.read_csv(raw / f"{split}_concepts_manual.csv"), 2).rename(columns={"value": "manual_cuis"})
    df = cap.merge(con, on="ID", how="left").merge(man, on="ID", how="left")
    df[["cuis", "manual_cuis"]] = df[["cuis", "manual_cuis"]].fillna("")

    def mods(s):
        found = [mod_cuis[c] for c in str(s).split(";") if c.strip() in mod_cuis]
        return sorted(set(found))

    m = df["manual_cuis"].map(mods)
    df["n_modalities"] = m.map(len)
    df["label"] = m.map(lambda x: x[0] if len(x) == 1 else ("multiple" if x else None))
    df["concept_names"] = df["cuis"].map(
        lambda s: "; ".join(cui_name.get(c.strip(), c.strip()) for c in str(s).split(";") if c.strip()))
    df["split"] = SPLIT_NAME[split]
    return df


def zip_index(zpath: Path) -> dict:
    idx = {}
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            stem = Path(info.filename).stem
            if stem.startswith("._"):  # macOS resource forks
                continue
            idx[stem] = info.filename
    return idx


def extract_resized(zpath: Path, members: dict, ids, out_dir: Path) -> list[str]:
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with zipfile.ZipFile(zpath) as z:
        for k, i in enumerate(ids):
            dest = out_dir / f"{i}.jpg"
            if not dest.exists():
                with z.open(members[i]) as f:
                    im = Image.open(io.BytesIO(f.read()))
                    im = im.convert("RGB")
                    w, h = im.size
                    s = SHORT_SIDE / min(w, h)
                    if s < 1:
                        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
                    im.save(dest, "JPEG", quality=90)
            paths.append(str(dest.relative_to(out_dir.parent.parent)))
            if (k + 1) % 2000 == 0 or k + 1 == len(ids):
                print(f"    {out_dir.name}: {k + 1}/{len(ids)} images")
    return paths


def write_data_md(folder: Path, obs: pd.DataFrame, avail: dict, mod_table: pd.DataFrame) -> None:
    cap_len = obs["caption"].str.split().map(len)
    md = f"""# ROCOv2 — radiology images + captions

**Samples:** {len(obs)} figure images from open-access biomedical papers
(PubMed Central), each paired with its **figure caption**, so row *i* is one
image–text pair (paired multimodal data).

| modality | kind | stored as |
|---|---|---|
| `image` | image | JPEG, RGB, shorter side resized to {SHORT_SIDE} px (aspect kept), path in column `image_path` |
| `text` | text | raw caption in column `caption` (median {int(cap_len.median())} words, 90th pct {int(cap_len.quantile(.9))}) |

Most images are grayscale scans stored as RGB. Captions are written by paper
authors for readers, not clinical reports: they often name the imaging
modality, body part and finding, and can mention arrows or panel letters.

## Labels (for evaluation)

- `label`: imaging modality, from ROCOv2's manually curated concepts
  (checked by a radiologist). Images with no or several modality concepts were
  dropped (see below).
- `cuis` / `concept_names`: all automatically extracted UMLS concepts of the
  caption (multi-label; usable as extra evaluation targets).

{label_table(obs, 'label')}

## Splits and subset

ROCOv2's own split (train / valid → `val` / test) is kept, so `test` images
come from different papers than `train`. A fixed random subset (seed {SEED})
of each split is used to keep training laptop-sized:

| split | available with one modality label | kept |
|---|---:|---:|
""" + "\n".join(f"| {s} | {avail[s]} | {int((obs['split'] == s).sum())} |" for s in ("train", "val", "test")) + f"""

Fit every preprocessing step and model on `train` only; use `val` for model
choices and `test` for the final numbers.

## Modality concepts used for the label

{df_to_md(mod_table)}

## Preparation already applied (not the agent's choice)

- Resizing (shorter side {SHORT_SIDE} px) and RGB conversion only; no cropping,
  no normalisation, no caption cleaning or tokenisation.

Source: {RECORD} (CC BY-NC 4.0; non-commercial use only; individual images
keep the licences listed in ROCOv2's `license_information.csv`).
"""
    (folder / "DATA.md").write_text(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=HERE / "raw")
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-val", type=int, default=2000)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    raw, out = args.raw_dir, args.out
    out.mkdir(parents=True, exist_ok=True)

    print("1/4 download (skips files already present)")
    for f in FILES:
        download(f"{RECORD}/files/{f}?download=1", raw / f)

    print("2/4 modality labels from the manual concepts")
    cm = pd.read_csv(raw / "cui_mapping.csv").iloc[:, :2]
    cm.columns = ["CUI", "name"]
    cui_name = dict(zip(cm["CUI"].astype(str).str.strip(), cm["name"].astype(str)))
    manual = set()
    for s in ("train", "valid", "test"):
        man = _cols(pd.read_csv(raw / f"{s}_concepts_manual.csv"), 2)
        for v in man["value"].dropna():
            manual.update(c.strip() for c in str(v).split(";") if c.strip())
    mod_cuis = {c: modality_of(cui_name.get(c, "")) for c in manual}
    mod_cuis = {c: m for c, m in mod_cuis.items() if m}
    if not mod_cuis:
        raise SystemExit("No imaging-modality concepts recognised in the manual concept files; "
                         "check cui_mapping.csv / MODALITY_PATTERNS.")

    frames = [load_split(raw, s, cui_name, mod_cuis) for s in ("train", "valid", "test")]
    df = pd.concat(frames, ignore_index=True)
    counts = Counter(c for v in df["manual_cuis"] for c in str(v).split(";") if c.strip() in mod_cuis)
    mod_table = pd.DataFrame([{"CUI": c, "concept": cui_name.get(c, ""), "label": m, "images": counts.get(c, 0)}
                              for c, m in sorted(mod_cuis.items(), key=lambda kv: -counts.get(kv[0], 0))])
    print(mod_table.to_string(index=False))
    n0 = len(df)
    print(f"  images: {n0}; no modality concept: {(df['n_modalities'] == 0).sum()}; "
          f"several: {(df['n_modalities'] > 1).sum()}")
    df = df[df["n_modalities"] == 1].copy()
    df = df[df["caption"].astype(str).str.strip().str.len() > 0]

    print("3/4 subset + extract images")
    rng = np.random.default_rng(SEED)
    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    keep, avail = [], {}
    for raw_split, split in SPLIT_NAME.items():
        part = df[df["split"] == split]
        idx = zip_index(raw / f"{raw_split}_images.zip")
        part = part[part["ID"].isin(idx)]
        avail[split] = len(part)
        n = caps[split] if caps[split] > 0 else len(part)
        if n < len(part):
            part = part.iloc[np.sort(rng.choice(len(part), size=n, replace=False))]
        part = part.copy()
        part["image_path"] = extract_resized(raw / f"{raw_split}_images.zip", idx, part["ID"].tolist(),
                                             out / "images" / split)
        keep.append(part)
    obs = pd.concat(keep, ignore_index=True)

    print("4/4 write")
    obs = obs.rename(columns={"ID": "sample_id"})
    obs["caption"] = obs["caption"].astype(str).str.strip()
    obs = obs[["sample_id", "split", "label", "image_path", "caption", "cuis", "concept_names"]]
    manifest = {
        "name": "rocov2",
        "description": "ROCOv2: radiology figure images paired with their captions (PMC open access)",
        "pairing": "each image paired with its own caption (row i of obs.csv = one image-text pair)",
        "obs": "obs.csv", "id_column": "sample_id", "split_column": "split",
        "label_columns": ["label"], "primary_label": "label",
        "multilabel_columns": ["cuis"],
        "modalities": [
            {"name": "image", "kind": "image", "column": "image_path",
             "description": f"radiology image, RGB JPEG, shorter side {SHORT_SIDE} px"},
            {"name": "text", "kind": "text", "column": "caption",
             "description": "figure caption written by the paper's authors"},
        ],
        "subset": {"seed": SEED, "max": caps, "available": avail},
        "source": {"url": RECORD, "license": "CC BY-NC 4.0",
                   "citation": "Rückert et al. (2024) ROCOv2. Scientific Data 11:688"},
    }
    write_manifest(out, manifest, obs)
    write_data_md(out, obs, avail, mod_table)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
