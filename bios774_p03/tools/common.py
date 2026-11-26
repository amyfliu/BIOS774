"""Shared helpers for the project-3 tools (dataset-agnostic; no dataset-specific branches).

Datasets follow data/FORMAT.md: dataset.json + obs.csv + one entry per modality
(kind = counts | image | text). Preprocessed feature matrices live in
<run>/features/<modality>__<spec>.npz and are what every learning method reads,
so methods never care whether a modality started as counts, pixels or words.
"""
from __future__ import annotations

import json
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- device / seeds
def get_device(prefer: str | None = None):
    """mps (Apple GPU) > cuda > cpu. Override with --device or MMAGENT_DEVICE."""
    import torch

    prefer = prefer or os.environ.get("MMAGENT_DEVICE")
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:  # only seed torch if a tool actually uses it
        sys.modules["torch"].manual_seed(seed)


# ----------------------------------------------------------------------------- json
def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rounded(x, nd=4):
    """Round floats recursively to nd SIGNIFICANT digits (compact files that still
    keep small values such as a learning rate of 1e-5 or a p-value)."""
    if isinstance(x, dict):
        return {k: rounded(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(f"{float(x):.{nd}g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


# ----------------------------------------------------------------------------- dataset
class Dataset:
    """Generic view of a prepared multimodal dataset folder (data/FORMAT.md)."""

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.meta = load_json(self.dir / "dataset.json")
        self.name = self.meta.get("name", self.dir.name)
        self.obs = pd.read_csv(self.dir / self.meta["obs"], low_memory=False)
        self.split = self.obs[self.meta.get("split_column", "split")].astype(str).values
        self.ids = self.obs[self.meta.get("id_column", "sample_id")].astype(str).values
        self.modalities = {m["name"]: m for m in self.meta["modalities"]}
        self.label_columns = list(self.meta.get("label_columns", []))
        self.primary_label = self.meta.get("primary_label", self.label_columns[0] if self.label_columns else None)

    def __len__(self):
        return len(self.obs)

    # ---- splits / labels
    def idx(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def labels(self, column: str | None = None):
        """(codes 0..K-1, class names) for a label column (default: primary)."""
        column = column or self.primary_label
        if column is None or column not in self.obs:
            return None, []
        names, codes = np.unique(self.obs[column].astype(str).values, return_inverse=True)
        return codes, list(names)

    # ---- raw modality access
    def kind(self, mod: str) -> str:
        return self.modalities[mod]["kind"]

    def counts(self, mod: str):
        import scipy.sparse as sp

        m = self.modalities[mod]
        assert m["kind"] == "counts", f"{mod} is {m['kind']}, not counts"
        return sp.load_npz(self.dir / m["file"]).tocsr()

    def features(self, mod: str) -> pd.DataFrame:
        m = self.modalities[mod]
        return pd.read_csv(self.dir / m["features"], sep="\t", low_memory=False)

    def texts(self, mod: str) -> list[str]:
        m = self.modalities[mod]
        assert m["kind"] == "text"
        return self.obs[m["column"]].fillna("").astype(str).tolist()

    def image_paths(self, mod: str) -> list[Path]:
        m = self.modalities[mod]
        assert m["kind"] == "image"
        return [self.dir / p for p in self.obs[m["column"]].astype(str)]


# ----------------------------------------------------------------------------- feature files
def feature_path(run_dir, mod: str, spec: str) -> Path:
    return Path(run_dir) / "features" / f"{mod}__{spec}.npz"


def save_features(path, X: np.ndarray, info: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32))
    save_json(info, path.with_suffix(".json"))


def load_features(path):
    """-> (X float32 [n_samples, d], info dict). path: .npz written by preprocess.py."""
    path = Path(path)
    X = np.load(path)["X"]
    info = load_json(path.with_suffix(".json")) if path.with_suffix(".json").exists() else {}
    return X, info


def load_embeddings(run_dir):
    """-> dict name -> array [n_samples, d] from <run>/embeddings.npz (+ run_config)."""
    d = np.load(Path(run_dir) / "embeddings.npz")
    return {k: d[k] for k in d.files}


def l2n(z: np.ndarray) -> np.ndarray:
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
