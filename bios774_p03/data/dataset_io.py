"""Shared helpers for the prepare.py scripts: downloading, stratified splits,
and writing/validating the prepared format described in data/FORMAT.md.

Only data preparation uses this module; the agent's tools load datasets via
tools/common.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# downloading
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, retries: int = 3) -> Path:
    """Download url -> dest with a progress line. Skips if dest already exists.

    Writes to dest.part first, so an interrupted download is never mistaken
    for a finished one.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already downloaded")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"  downloading {dest.name}  ({url})")
            req = urllib.request.Request(url, headers={"User-Agent": "bios774-p03/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done, t0, last = 0, time.time(), 0.0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last > 1 or done == total:
                        last = now
                        mb = done / 1e6
                        rate = mb / max(now - t0, 1e-6)
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        sys.stdout.write(f"\r    {pct} {mb:8.1f} MB  {rate:6.1f} MB/s")
                        sys.stdout.flush()
            sys.stdout.write("\n")
            if total and tmp.stat().st_size != total:
                raise IOError(f"incomplete download ({tmp.stat().st_size} of {total} bytes)")
            os.replace(tmp, dest)
            return dest
        except Exception as e:  # noqa: BLE001 - report and retry any failure
            print(f"\n    attempt {attempt} failed: {e}")
            if attempt == retries:
                raise SystemExit(
                    f"Could not download {url}.\n"
                    f"Download it manually (e.g. in a browser) and save it as:\n  {dest}\n"
                    f"then re-run this script."
                )
            time.sleep(5 * attempt)
    return dest


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def stratified_split(labels, frac=(0.7, 0.15, 0.15), seed: int = 0) -> np.ndarray:
    """Return an array of 'train'/'val'/'test' per sample, stratified by label.

    Classes too small to split (< 3 samples) go entirely to train.
    """
    labels = np.asarray(labels).astype(str)
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=object)
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        rng.shuffle(idx)
        n = len(idx)
        if n < 3:
            out[idx] = "train"
            continue
        n_val = max(1, int(round(frac[1] * n)))
        n_test = max(1, int(round(frac[2] * n)))
        n_train = n - n_val - n_test
        out[idx[:n_train]] = "train"
        out[idx[n_train:n_train + n_val]] = "val"
        out[idx[n_train + n_val:]] = "test"
    return out


# --------------------------------------------------------------------------- #
# writing + validating
# --------------------------------------------------------------------------- #
def write_manifest(folder: Path, manifest: dict, obs: pd.DataFrame) -> None:
    folder = Path(folder)
    obs.to_csv(folder / manifest.get("obs", "obs.csv"), index=False)
    split_col = manifest.get("split_column", "split")
    manifest["n_samples"] = int(len(obs))
    manifest["splits"] = {s: int((obs[split_col] == s).sum()) for s in SPLITS}
    with open(folder / "dataset.json", "w") as f:
        json.dump(manifest, f, indent=2)
    validate(folder)


def validate(folder: Path) -> dict:
    """Check a prepared dataset folder against data/FORMAT.md. Raises on error."""
    import scipy.sparse as sp

    folder = Path(folder)
    with open(folder / "dataset.json") as f:
        m = json.load(f)
    obs = pd.read_csv(folder / m["obs"])
    n = len(obs)
    assert n == m["n_samples"], "n_samples does not match obs.csv"
    assert m["id_column"] in obs and obs[m["id_column"]].is_unique, "sample ids missing or not unique"
    assert set(obs[m["split_column"]].unique()) <= set(SPLITS), "unknown split values"
    for col in m.get("label_columns", []):
        assert col in obs, f"label column {col} missing"
    assert len(m["modalities"]) >= 2, "a multimodal dataset needs >= 2 modalities"
    for mod in m["modalities"]:
        kind = mod["kind"]
        if kind == "counts":
            X = sp.load_npz(folder / mod["file"])
            assert X.shape[0] == n, f"{mod['name']}: {X.shape[0]} rows, obs has {n}"
            feats = pd.read_csv(folder / mod["features"], sep="\t")
            assert len(feats) == X.shape[1], f"{mod['name']}: feature table does not match columns"
        elif kind == "image":
            col = mod["column"]
            assert col in obs, f"{mod['name']}: column {col} missing"
            missing = [p for p in obs[col].head(50) if not (folder / p).exists()]
            assert not missing, f"{mod['name']}: missing image files, e.g. {missing[:3]}"
        elif kind == "text":
            col = mod["column"]
            assert col in obs, f"{mod['name']}: column {col} missing"
            assert obs[col].notna().all(), f"{mod['name']}: empty text entries"
        else:
            raise AssertionError(f"unknown modality kind {kind}")
    print(f"  [ok] {folder.name}: {n} samples, modalities = "
          + ", ".join(f"{x['name']}({x['kind']})" for x in m["modalities"])
          + f", splits = {m['splits']}")
    return m


def df_to_md(df: pd.DataFrame) -> str:
    """Plain markdown table (no extra dependency)."""
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def label_table(obs: pd.DataFrame, label: str, split_col: str = "split") -> str:
    """Markdown table of label counts per split (for DATA.md)."""
    tab = pd.crosstab(obs[label], obs[split_col]).reindex(columns=list(SPLITS), fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total", ascending=False)
    lines = [f"| {label} | " + " | ".join(tab.columns) + " |",
             "|---|" + "---:|" * len(tab.columns)]
    for lab, row in tab.iterrows():
        lines.append(f"| {lab} | " + " | ".join(str(int(v)) for v in row.values) + " |")
    return "\n".join(lines)
