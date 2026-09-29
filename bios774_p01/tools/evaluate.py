"""
evaluate.py — dataset-agnostic embedding evaluation.

Usage:
    python tools/evaluate.py --data data/<dataset>/prepared.npz \
        --embedding outputs/<dataset>/pca_2d/embedding.npy \
        --out outputs/<dataset>/pca_2d/metrics.json

Computes, where applicable:
    trustworthiness         (neighborhood preservation, high-D -> low-D)
    reconstruction metrics  (for PCA-like methods only, via run_config.json if present)
    silhouette_score        (only if labels exist in the data — quality of
                              cluster separation in the embedding, NOT a DR
                              correctness metric by itself)

If run_dr.py was called with --subsample, this script auto-detects the
sample_indices.npy it wrote next to --embedding and evaluates against the
same rows (override with --subsample_indices). Independently, trustworthiness
and silhouette are each further subsampled internally above their own
--*_sample_size threshold, since both are O(n^2) in memory/time.

This script makes no judgment about whether a result is "good enough" —
it reports numbers. The agent interprets them against the task's goals.
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preprocessing", default="standardize",
                    help="Must match what run_dr.py used, for a fair trustworthiness comparison.")
    ap.add_argument("--silhouette_sample_size", type=int, default=5000,
                    help="Silhouette is O(n^2); above this many samples, both silhouettes "
                         "are computed on the same random subsample.")
    ap.add_argument("--trustworthiness_sample_size", type=int, default=5000,
                    help="trustworthiness() builds a dense n x n distance matrix; above this "
                         "many samples, it is computed on a random subsample of the (already "
                         "subsampled, if applicable) rows instead.")
    ap.add_argument("--subsample_indices", default=None,
                    help="Path to sample_indices.npy written by run_dr.py's --subsample. If "
                         "omitted, auto-detected as sample_indices.npy next to --embedding.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(float)
    y = d["y"] if "y" in d else None
    Z = np.load(args.embedding)

    subsample_indices_path = args.subsample_indices or os.path.join(
        os.path.dirname(args.embedding), "sample_indices.npy"
    )
    subsampled = os.path.exists(subsample_indices_path)
    if subsampled:
        sample_indices = np.load(subsample_indices_path)
        X = X[sample_indices]
        if y is not None:
            y = y[sample_indices]

    from sklearn.preprocessing import StandardScaler
    if args.preprocessing in ("standardize", "log1p_standardize"):
        Xp = np.log1p(np.clip(X, 0, None)) if args.preprocessing == "log1p_standardize" else X
        Xp = StandardScaler().fit_transform(Xp)
    elif args.preprocessing == "log1p":
        Xp = np.log1p(np.clip(X, 0, None))
    else:
        Xp = X

    metrics = {}

    if Z.shape[0] != Xp.shape[0]:
        metrics["error"] = (
            f"Embedding has {Z.shape[0]} rows but data has {Xp.shape[0]}; rows must "
            "correspond one-to-one. If run_dr.py subsampled, evaluate against the "
            "same subsample."
        )
        _write(args.out, metrics)
        return

    # Trustworthiness: does the embedding preserve high-D nearest neighbors?
    # trustworthiness() builds a dense n x n distance matrix internally, so
    # above --trustworthiness_sample_size it is computed on a random subsample
    # of rows instead (same subsample used for both spaces).
    try:
        from sklearn.manifold import trustworthiness
        n = Xp.shape[0]
        if n > args.trustworthiness_sample_size:
            rng = np.random.default_rng(args.seed)
            t_idx = rng.choice(n, size=args.trustworthiness_sample_size, replace=False)
        else:
            t_idx = np.arange(n)
        n_neighbors = min(15, len(t_idx) // 10 or 1)
        metrics["trustworthiness"] = float(
            trustworthiness(Xp[t_idx], Z[t_idx], n_neighbors=n_neighbors)
        )
        metrics["trustworthiness_n_neighbors"] = n_neighbors
        metrics["trustworthiness_n_points"] = int(len(t_idx))
    except Exception as e:
        metrics["trustworthiness_error"] = str(e)

    # Silhouette on labels, computed twice on the SAME points:
    #   - in the embedding Z (how separated the groups look in the plot)
    #   - in the preprocessed input space (how separated they actually are
    #     before dimension reduction)
    # The gap between the two is the part of the visual separation that the
    # method added or removed.
    if y is not None:
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(y)) > 1:
                n = Xp.shape[0]
                if n > args.silhouette_sample_size:
                    rng = np.random.default_rng(args.seed)
                    idx = rng.choice(n, size=args.silhouette_sample_size, replace=False)
                else:
                    idx = np.arange(n)
                if len(np.unique(y[idx])) > 1:
                    s_emb = float(silhouette_score(Z[idx], y[idx]))
                    s_in = float(silhouette_score(Xp[idx], y[idx]))
                    metrics["silhouette_score_on_labels"] = s_emb
                    metrics["silhouette_input_space"] = s_in
                    metrics["silhouette_gap_embedding_minus_input"] = s_emb - s_in
                    metrics["silhouette_n_points"] = int(len(idx))
                    metrics["silhouette_input_space_dims"] = int(Xp.shape[1])
        except Exception as e:
            metrics["silhouette_error"] = str(e)

    metrics["embedding_shape"] = list(Z.shape)
    metrics["n_samples"] = int(Xp.shape[0])
    metrics["evaluated_against_subsample"] = subsampled
    if subsampled:
        metrics["subsample_indices_file"] = subsample_indices_path
    _write(args.out, metrics)


def _write(path, metrics):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()