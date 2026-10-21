"""Shared helpers for the project-2 tools (dataset-agnostic; no dataset-specific branches)."""
from __future__ import annotations

import json
import os
import warnings
import pickle
import random
import time
from pathlib import Path

import numpy as np

# MPS: let any op without a Metal kernel fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402


# NumPy 2.0 + Apple Accelerate raises spurious floating-point flags in matmul
# ("divide by zero / overflow / invalid value encountered in matmul") even when the
# result is finite and correct. Silence only those messages; real NaNs still surface.
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

# ----------------------------------------------------------------------------- device / seeds
def get_device(prefer: str | None = None) -> torch.device:
    """mps (Apple GPU) > cuda > cpu. Override with --device or GENAGENT_DEVICE."""
    prefer = prefer or os.environ.get("GENAGENT_DEVICE")
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
    torch.manual_seed(seed)


def to_tensor(a: np.ndarray, device) -> torch.Tensor:
    # float32 everywhere: MPS has no float64 support.
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)


# ----------------------------------------------------------------------------- dataset
class Dataset:
    """Standardized view of data/<name>/prepared.npz."""

    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.name = self.dir.name
        d = np.load(self.dir / "prepared.npz", allow_pickle=True)
        self.X = d["X"]
        self.y_raw = d["y"] if "y" in d else None
        self.split = d["split"] if "split" in d else np.full(len(self.X), "train")
        self.image_shape = tuple(int(v) for v in d["image_shape"]) if "image_shape" in d else None
        self.feature_names = d["feature_names"] if "feature_names" in d else None
        self.label_values = d["label_values"] if "label_values" in d else None
        self.label_names_arr = d["label_names"] if "label_names" in d else None
        # Encode labels to 0..K-1 (sorted unique), keep human-readable names.
        if self.y_raw is not None:
            self.classes, self.y = np.unique(self.y_raw, return_inverse=True)
        else:
            self.classes, self.y = np.array([]), None

    @property
    def modality(self) -> str:
        return "image" if self.image_shape is not None else "tabular"

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def class_name(self, k: int) -> str:
        raw = self.classes[k]
        if self.label_values is not None and self.label_names_arr is not None:
            hit = np.where(self.label_values.astype(str) == str(raw))[0]
            if len(hit):
                return str(self.label_names_arr[hit[0]])
        return str(raw)

    def class_names(self) -> list[str]:
        return [self.class_name(k) for k in range(self.n_classes)]

    def resolve_classes(self, spec: str | None) -> list[int]:
        """'0,3' or 'B cells,Megakaryocytes' -> encoded class indices."""
        if not spec:
            return []
        names = self.class_names()
        out = []
        for tok in [t.strip() for t in spec.split(",") if t.strip()]:
            if tok in names:
                out.append(names.index(tok))
            elif tok in [str(c) for c in self.classes]:
                out.append([str(c) for c in self.classes].index(tok))
            elif tok.isdigit() and int(tok) < self.n_classes:
                out.append(int(tok))
            else:
                raise ValueError(f"Unknown class '{tok}'. Known: {names}")
        return out

    def indices(self, split: str, include=None, exclude=None, max_n=None, seed=0) -> np.ndarray:
        idx = np.where(self.split == split)[0]
        if self.y is not None and include:
            idx = idx[np.isin(self.y[idx], include)]
        if self.y is not None and exclude:
            idx = idx[~np.isin(self.y[idx], exclude)]
        if max_n and len(idx) > max_n:
            idx = stratified_subsample(idx, self.y, max_n, seed)
        return idx


def stratified_subsample(idx, y, n, seed=0):
    rng = np.random.default_rng(seed)
    if y is None:
        return np.sort(rng.choice(idx, n, replace=False))
    yy = y[idx]
    out = []
    for c in np.unique(yy):
        ci = idx[yy == c]
        k = max(1, int(round(n * len(ci) / len(idx))))
        out.append(rng.choice(ci, min(k, len(ci)), replace=False))
    return np.sort(np.concatenate(out))


# ----------------------------------------------------------------------------- preprocessing
class Prep:
    """Fit-on-train preprocessing pipeline from a spec string such as
    'libnorm,log1p,hvg:2000,standardize' (tabular) or 'scale_pm1' (image).

    Steps
      libnorm[:T]   scale each row to total T (default 1e4)       (tabular)
      log1p         x -> log(1+x)
      hvg:N         keep the N highest-variance features (train)  (tabular)
      standardize   z-score each feature with train mean/sd
      pca:K         project onto top-K principal components (train)
      scale01       x / 255                                        (image)
      scale_pm1     x / 127.5 - 1                                  (image)
      none          identity
    """

    def __init__(self, spec: str):
        self.spec = spec or "none"
        self.steps = [s.strip() for s in self.spec.split(",") if s.strip()]
        self.state: dict = {}

    def fit_transform(self, X):
        return self._run(X.astype(np.float32), fit=True)

    def transform(self, X):
        return self._run(X.astype(np.float32), fit=False)

    def _run(self, X, fit):
        for i, step in enumerate(self.steps):
            name, _, arg = step.partition(":")
            key = f"{i}_{name}"
            if name == "none":
                continue
            elif name == "libnorm":
                T = float(arg) if arg else 1e4
                X = X / np.maximum(X.sum(1, keepdims=True), 1e-8) * T
            elif name == "log1p":
                X = np.log1p(np.maximum(X, 0))
            elif name == "hvg":
                if fit:
                    self.state[key] = np.sort(np.argsort(X.var(0))[::-1][: int(arg)])
                X = X[:, self.state[key]]
            elif name == "standardize":
                if fit:
                    self.state[key] = (X.mean(0), X.std(0) + 1e-6)
                m, s = self.state[key]
                X = (X - m) / s
            elif name == "pca":
                if fit:
                    mu = X.mean(0)
                    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
                    k = int(arg)
                    self.state[key] = (mu, Vt[:k].astype(np.float32),
                                       float((S[:k] ** 2).sum() / (S ** 2).sum()))
                mu, V, _ = self.state[key]
                X = (X - mu) @ V.T
            elif name == "scale01":
                X = X / 255.0
            elif name == "scale_pm1":
                X = X / 127.5 - 1.0
            else:
                raise ValueError(f"Unknown preprocessing step '{step}'")
        return X.astype(np.float32)

    def inverse_image(self, Z):
        """Model space -> uint8 pixels (image pipelines only)."""
        X = Z.astype(np.float32)
        for step in reversed(self.steps):
            if step == "scale01":
                X = X * 255.0
            elif step == "scale_pm1":
                X = (X + 1.0) * 127.5
        return np.clip(np.round(X), 0, 255).astype(np.uint8)

    def info(self) -> dict:
        out = {"spec": self.spec}
        for k, v in self.state.items():
            if k.split("_", 1)[1] == "pca":
                out["pca_explained_variance"] = round(v[2], 4)
            if k.split("_", 1)[1] == "hvg":
                out["n_features_kept"] = int(len(v))
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"spec": self.spec, "state": self.state}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        p = cls(d["spec"])
        p.state = d["state"]
        return p


def to_model_shape(X: np.ndarray, image_shape, backbone: str) -> np.ndarray:
    """Flat rows -> (N, C, H, W) for conv backbones on image data."""
    if image_shape is not None and backbone == "conv":
        H, W, C = image_shape
        return X.reshape(-1, H, W, C).transpose(0, 3, 1, 2).copy()
    return X


def to_flat(X: np.ndarray, image_shape) -> np.ndarray:
    """(N, C, H, W) -> flat HWC rows (inverse of to_model_shape)."""
    if X.ndim == 4:
        return X.transpose(0, 2, 3, 1).reshape(len(X), -1)
    return X


# ----------------------------------------------------------------------------- io
def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Timer:
    def __init__(self, max_minutes: float | None):
        self.t0 = time.time()
        self.max_s = max_minutes * 60 if max_minutes else None

    @property
    def elapsed(self):
        return time.time() - self.t0

    def out_of_time(self):
        return self.max_s is not None and self.elapsed > self.max_s
