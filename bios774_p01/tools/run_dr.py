"""
run_dr.py — dataset-agnostic dimension reduction runner.

Usage:
    python tools/run_dr.py --data data/<dataset>/prepared.npz \
        --method pca --n_components 2 \
        --preprocessing standardize --out_dir outputs/<dataset>/pca_2d \
        [--params '{"perplexity": 30}']

Supported --method values:
    pca, kernel_pca, mds, isomap, lle, laplacian_eigenmaps, tsne, umap

Supported --preprocessing values:
    none, standardize, log1p, log1p_standardize

Optional --subsample N [--seed S]:
    Uniform random subsample without replacement, drawn BEFORE preprocessing
    (so a data-dependent preprocessing mode like standardize is fit on the
    subsample, not the full dataset). Use for methods that don't scale to a
    dataset's full n_samples (e.g. MDS, Kernel PCA, Isomap — all build a
    dense n x n matrix internally). Row indices are saved to
    sample_indices.npy in out_dir; evaluate.py auto-detects that file and
    evaluates against the same rows.

Writes to out_dir:
    embedding.npy       (n_samples, n_components)
    plot.png            (first 2 dims, colored by label if available)
    run_config.json     (exact method/params/preprocessing used — for reproducibility)
    sample_indices.npy  (only if --subsample was used)

Optional standardized .npz fields `label_values` + `label_names`:
    If a dataset's prepare.py provides these (a class code, e.g. matching y's
    values, alongside its display name, e.g. a tissue type), plot.png's
    colorbar shows the display names instead of raw codes. Both are absent
    for most datasets — coloring then falls back to y's raw values, as
    before. This script never assumes they exist.

This script performs NO decision-making about which method/params/preprocessing
to use — those are arguments supplied by the agent based on inspect_data.py's
output and the method-selection skill. This script only executes.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Conventional axis-label prefixes per method (e.g. "PC1"/"PC2", not
# "PCA_1"/"PCA_2"). Anything not listed falls back to the method name
# upper-cased, so a new --method value still gets a sane label rather than
# an error.
AXIS_LABEL_PREFIX = {
    "pca": "PC",
    "kernel_pca": "KPC",
    "mds": "MDS",
    "isomap": "Isomap",
    "lle": "LLE",
    "laplacian_eigenmaps": "LE",
    "diffusion_map": "DC",
    "tsne": "tSNE",
    "umap": "UMAP",
}

# Full display names for plot titles (e.g. "t-SNE", not "tSNE" or "TSNE").
# Anything not listed falls back to a generic title-cased version of the
# method name, so a new --method value still gets a sane title.
METHOD_DISPLAY_NAME = {
    "pca": "PCA",
    "kernel_pca": "Kernel PCA",
    "mds": "MDS",
    "isomap": "Isomap",
    "lle": "LLE",
    "laplacian_eigenmaps": "Laplacian Eigenmaps",
    "diffusion_map": "Diffusion Map",
    "tsne": "t-SNE",
    "umap": "UMAP",
}


def apply_preprocessing(X, mode):
    if mode == "none":
        return X
    if mode == "log1p":
        return np.log1p(np.clip(X, a_min=0, a_max=None))
    if mode in ("standardize", "log1p_standardize"):
        from sklearn.preprocessing import StandardScaler
        Xp = np.log1p(np.clip(X, a_min=0, a_max=None)) if mode == "log1p_standardize" else X
        return StandardScaler().fit_transform(Xp)
    raise ValueError(f"Unknown preprocessing mode: {mode}")


def get_model(method, n_components, params):
    if params is None:
        params = {}
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, **params)
    if method == "kernel_pca":
        from sklearn.decomposition import KernelPCA
        params.setdefault("kernel", "rbf")
        return KernelPCA(n_components=n_components, **params)
    if method == "mds":
        from sklearn.manifold import MDS
        params.setdefault("normalized_stress", "auto")
        return MDS(n_components=n_components, **params)
    if method == "isomap":
        from sklearn.manifold import Isomap
        params.setdefault("n_neighbors", 15)
        return Isomap(n_components=n_components, **params)
    if method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding
        params.setdefault("n_neighbors", 15)
        return LocallyLinearEmbedding(n_components=n_components, **params)
    if method == "laplacian_eigenmaps":
        from sklearn.manifold import SpectralEmbedding
        params.setdefault("n_neighbors", 15)
        return SpectralEmbedding(n_components=n_components, **params)
    if method == "diffusion_map":
        from diffusion_map import DiffusionMap
        params.setdefault("n_neighbors", 15)
        return DiffusionMap(n_components=n_components, **params)
    if method == "tsne":
        from sklearn.manifold import TSNE
        params.setdefault("perplexity", 30)
        params.setdefault("init", "pca")
        return TSNE(n_components=n_components, **params)
    if method == "umap":
        import umap
        params.setdefault("n_neighbors", 15)
        params.setdefault("min_dist", 0.1)
        return umap.UMAP(n_components=n_components, **params)
    raise ValueError(f"Unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--n_components", type=int, default=2)
    ap.add_argument("--preprocessing", default="standardize")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subsample", type=int, default=None,
                     help="Randomly subsample to this many rows (without replacement) "
                          "before preprocessing/fitting.")
    ap.add_argument("--seed", type=int, default=0,
                     help="Random seed for --subsample.")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(float)
    y = d["y"] if "y" in d else None

    sample_indices = None
    if args.subsample is not None and args.subsample < X.shape[0]:
        rng = np.random.default_rng(args.seed)
        sample_indices = np.sort(rng.choice(X.shape[0], size=args.subsample, replace=False))
        X = X[sample_indices]
        if y is not None:
            y = y[sample_indices]

    Xp = apply_preprocessing(X, args.preprocessing)
    params = json.loads(args.params)
    model = get_model(args.method, args.n_components, params)
    Z = model.fit_transform(Xp)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embedding.npy"), Z)
    if sample_indices is not None:
        np.save(os.path.join(args.out_dir, "sample_indices.npy"), sample_indices)

    plt.figure(figsize=(6, 5))
    if y is not None:
        # y may be numeric or string-encoded categorical labels (e.g. "0", "1", ...);
        # np.unique maps either to integer codes so coloring works regardless of
        # storage dtype.
        code_order, codes = np.unique(y, return_inverse=True)
        # Optional standardized fields: label_values (unique codes, any dtype) +
        # label_names (matching display strings) let a dataset's prepare.py attach
        # human-readable names (e.g. tissue types) without run_dr.py knowing
        # anything dataset-specific. Absent for most datasets -> falls back to
        # the raw codes, exactly as before.
        tick_labels = [str(v) for v in code_order]
        if "label_names" in d and "label_values" in d:
            name_lookup = dict(zip(d["label_values"].astype(str), d["label_names"].astype(str)))
            tick_labels = [name_lookup.get(str(v), str(v)) for v in code_order]
        sc = plt.scatter(Z[:, 0], Z[:, 1], c=codes, cmap="tab10", s=8, alpha=0.7)
        cbar = plt.colorbar(sc, label="label")
        cbar.set_ticks(np.arange(len(code_order)))
        cbar.set_ticklabels(tick_labels)
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7)
    axis_prefix = AXIS_LABEL_PREFIX.get(args.method, args.method.upper())
    plt.xlabel(f"{axis_prefix}1")
    plt.ylabel(f"{axis_prefix}2")
    method_display = METHOD_DISPLAY_NAME.get(args.method, args.method.replace("_", " ").title())
    dataset_name = os.path.basename(os.path.dirname(args.data))
    plt.title(f"{method_display} Projection of {dataset_name} ({args.preprocessing})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "plot.png"), dpi=130)
    plt.close()

    config = {
        "data": args.data,
        "method": args.method,
        "n_components": args.n_components,
        "preprocessing": args.preprocessing,
        "params": params,
        "embedding_shape": list(Z.shape),
        "subsample": (
            {"n": int(args.subsample), "seed": args.seed, "indices_file": "sample_indices.npy"}
            if sample_indices is not None else None
        ),
    }
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
