"""
inspect_data.py — dataset-agnostic profiling tool.

Usage:
    python tools/inspect_data.py --data data/<dataset>/prepared.npz

Reads a standardized .npz with keys:
    X               (n_samples, n_features) float array, feature matrix
    feature_names   (n_features,) array of strings [optional]
    sample_ids      (n_samples,) array of strings [optional]
    y               (n_samples,) array, labels for evaluation only [optional]

Writes outputs/<dataset>/profile.json with characteristics the agent should
use to decide preprocessing and method choice: shape, missingness, sparsity,
feature scale heterogeneity, and a handful of structure hints (rank estimate,
duplicate rows, constant columns).

This script makes NO decisions about preprocessing or DR method — it only
reports facts. Decision-making belongs to the agent, not this tool.
"""
import argparse
import json
import os

import numpy as np


def profile(data_path: str) -> dict:
    d = np.load(data_path, allow_pickle=True)
    X = d["X"].astype(float)
    n_samples, n_features = X.shape

    n_missing = int(np.isnan(X).sum())
    missing_frac = n_missing / X.size if X.size else 0.0

    # sparsity: fraction of exact zeros (meaningful for count data like scRNA-seq)
    n_zero = int(np.sum(X == 0))
    sparsity = n_zero / X.size if X.size else 0.0

    finite = X[np.isfinite(X)]
    col_std = np.nanstd(X, axis=0)
    col_mean = np.nanmean(X, axis=0)
    n_constant_cols = int(np.sum(col_std == 0))

    # scale heterogeneity: ratio of max to min nonzero std across features
    nonzero_std = col_std[col_std > 0]
    scale_ratio = float(nonzero_std.max() / nonzero_std.min()) if len(nonzero_std) > 1 else 1.0

    n_dupe_rows = int(n_samples - len(np.unique(X, axis=0))) if n_samples * n_features < 5_000_000 else None

    # cheap rank estimate via a truncated SVD sample (skip if huge)
    rank_estimate = None
    if n_samples * n_features <= 5_000_000:
        try:
            s = np.linalg.svd(np.nan_to_num(X - col_mean), compute_uv=False)
            energy = np.cumsum(s ** 2) / np.sum(s ** 2)
            rank_estimate = {
                "dims_for_90pct_variance": int(np.searchsorted(energy, 0.90) + 1),
                "dims_for_50pct_variance": int(np.searchsorted(energy, 0.50) + 1),
                "n_singular_values": len(s),
            }
        except Exception as e:
            rank_estimate = {"error": str(e)}

    result = {
        "dataset": os.path.basename(os.path.dirname(data_path)),
        "n_samples": n_samples,
        "n_features": n_features,
        "missing_fraction": missing_frac,
        "sparsity_fraction": sparsity,
        "n_constant_columns": n_constant_cols,
        "feature_scale_ratio_max_over_min_std": scale_ratio,
        "n_duplicate_rows": n_dupe_rows,
        "has_labels": "y" in d,
        "n_label_classes": int(len(np.unique(d["y"]))) if "y" in d else None,
        "rank_estimate_from_svd": rank_estimate,
        "value_range": {"min": float(np.nanmin(X)), "max": float(np.nanmax(X))},
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = profile(args.data)
    dataset_dir = os.path.dirname(args.data)
    out_path = args.out or os.path.join("outputs", os.path.basename(dataset_dir), "profile.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
