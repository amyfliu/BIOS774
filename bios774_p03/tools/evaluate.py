"""evaluate.py — score one trained representation (all numbers on the TEST split).

  python tools/evaluate.py --data D --run O/<label> --tasks structure,retrieval,collapse \
      [--labels label,label_fine] [--cross rna->atac:binary_top5000,atac->rna:log1p_hvg2000]

Tasks (choose the ones that measure the objective):
  structure   Is the map meaningful? For each embedding (joint and per modality):
              linear probe (logistic regression fit on train) and 15-NN classifier
              -> balanced accuracy + macro-F1 on test; k-means with k = #classes
              -> ARI / NMI; cosine silhouette by label.
  retrieval   Do the modalities agree? Paired cross-modal retrieval on test:
              recall@1/5/10 both directions, median rank, FOSCTTM (fraction of
              samples closer than the true partner: 0 = perfect, 0.5 = random).
              Needs two modality embeddings of equal dimension.
  collapse    Health checks: effective rank, dead dimensions, alignment and
              uniformity (Wang & Isola 2020), modality gap.
  cross       Cross-modal prediction: ridge regression from this run's
              embedding of the SOURCE modality to a preprocessed TARGET matrix
              (--cross src->tgt:spec). Continuous targets: per-feature and
              per-sample Pearson r; binary targets: per-feature and per-sample
              AUROC. Always next to the train-mean baseline.

Each headline number comes with a chance level or baseline and a 95% bootstrap
interval over the evaluated samples.

--split val   for CHOOSING between models (writes metrics_val.json; no plots)
--split test  for the FINAL numbers of the chosen models (writes metrics.json +
              embedding.png). Never pick a model by its test numbers.
Compare runs with tools/compare_runs.py --split val|test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, feature_path, l2n, load_embeddings, load_features, load_json, rounded, save_json  # noqa: E402

RNG = np.random.default_rng(0)


def boot_ci(stat_fn, n, B=200):
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        try:
            vals.append(stat_fn(idx))
        except ValueError:
            continue
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ----------------------------------------------------------------------------- structure
def structure(Z, codes, classes, tr, te):
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, f1_score,
                                 normalized_mutual_info_score, silhouette_score)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Z[tr])
    A = sc.transform(Z)
    ytr, yte = codes[tr], codes[te]
    out = {"n_classes": len(classes), "majority_baseline_accuracy": float(np.bincount(yte).max() / len(yte)),
           "chance_balanced_accuracy": 1 / len(np.unique(ytr))}
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(A[tr], ytr)
    p = lr.predict(A[te])
    out["probe_balanced_accuracy"] = float(balanced_accuracy_score(yte, p))
    out["probe_balanced_accuracy_ci95"] = boot_ci(lambda i: balanced_accuracy_score(yte[i], p[i]), len(te))
    out["probe_macro_f1"] = float(f1_score(yte, p, average="macro"))
    out["probe_accuracy"] = float((p == yte).mean())
    per = {}
    for k, c in enumerate(classes):
        m = yte == k
        if m.sum():
            per[c] = {"n_test": int(m.sum()), "recall": float((p[m] == k).mean())}
    out["probe_per_class_recall"] = per
    knn = KNeighborsClassifier(n_neighbors=15, metric="cosine").fit(Z[tr], ytr)
    pk = knn.predict(Z[te])
    out["knn15_balanced_accuracy"] = float(balanced_accuracy_score(yte, pk))
    k = len(np.unique(yte))
    cl = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(A[te])
    out["kmeans_ari"] = float(adjusted_rand_score(yte, cl))
    out["kmeans_nmi"] = float(normalized_mutual_info_score(yte, cl))
    sub = te if len(te) <= 4000 else RNG.choice(te, 4000, replace=False)
    if len(np.unique(codes[sub])) > 1:
        out["silhouette_cosine"] = float(silhouette_score(Z[sub], codes[sub], metric="cosine"))
    return out


# ----------------------------------------------------------------------------- retrieval
def ranks(za, zb):
    """rank (0 = best) of the true partner of each row of za among all rows of zb."""
    s = l2n(za) @ l2n(zb).T
    true = np.diag(s)
    return (s > true[:, None]).sum(1)


def retrieval(Za, Zb, te, n_max=2000):
    idx = te if len(te) <= n_max else np.sort(RNG.choice(te, n_max, replace=False))
    n = len(idx)
    out = {"n_candidates": n}
    for name, (A, B) in {"a_to_b": (Za[idx], Zb[idx]), "b_to_a": (Zb[idx], Za[idx])}.items():
        r = ranks(A, B)
        d = {f"recall_at_{k}": float((r < k).mean()) for k in (1, 5, 10)}
        d["recall_at_5_ci95"] = boot_ci(lambda i: float((r[i] < 5).mean()), n)
        d["median_rank"] = float(np.median(r) + 1)
        d["foscttm"] = float((r / (n - 1)).mean())
        out[name] = d
    out["chance"] = {f"recall_at_{k}": k / n for k in (1, 5, 10)} | {"median_rank": (n + 1) / 2, "foscttm": 0.5}
    return out, idx


# ----------------------------------------------------------------------------- collapse
def collapse(Z):
    Zc = Z - Z.mean(0)
    s = np.linalg.svd(Zc, compute_uv=False)
    p = s ** 2 / max((s ** 2).sum(), 1e-12)
    p = p[p > 0]
    sd = Z.std(0)
    return {"dim": int(Z.shape[1]), "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
            "dead_dims": int((sd < 1e-3 * max(sd.max(), 1e-12)).sum()),
            "top_component_variance_share": float(p[0]) if len(p) else None}


def uniformity(Z, n=2000):
    Z = l2n(Z[:n] if len(Z) > n else Z)
    d2 = np.maximum(2 - 2 * Z @ Z.T, 0)
    iu = np.triu_indices(len(Z), 1)
    return float(np.log(np.mean(np.exp(-2 * d2[iu]))))


# ----------------------------------------------------------------------------- cross-modal prediction
def cross_predict(Zs, Y, tr, va, te, binary):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import roc_auc_score

    mu, sd = Zs[tr].mean(0), Zs[tr].std(0) + 1e-8
    A = (Zs - mu) / sd
    best = None
    for alpha in (0.1, 1, 10, 100, 1000):
        m = Ridge(alpha=alpha).fit(A[tr], Y[tr])
        err = float(((m.predict(A[va]) - Y[va]) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, alpha, m)
    _, alpha, m = best
    P = m.predict(A[te])
    T = Y[te]
    base = np.tile(Y[tr].mean(0), (len(te), 1))
    out = {"ridge_alpha_chosen_on_val": alpha, "n_targets": int(Y.shape[1]), "n_test": int(len(te))}

    def per_sample_r(P_, T_):
        Pc, Tc = P_ - P_.mean(1, keepdims=True), T_ - T_.mean(1, keepdims=True)
        den = np.sqrt((Pc ** 2).sum(1) * (Tc ** 2).sum(1))
        return np.where(den > 0, (Pc * Tc).sum(1) / np.maximum(den, 1e-12), 0)

    if binary:
        ok = (T.min(0) != T.max(0))
        aucs = np.array([roc_auc_score(T[:, j], P[:, j]) for j in np.flatnonzero(ok)])
        out["per_feature_auroc_mean"] = float(aucs.mean())
        out["per_feature_auroc_median"] = float(np.median(aucs))
        out["per_feature_auroc_baseline"] = 0.5
        ps = np.array([roc_auc_score(T[i], P[i]) for i in range(len(te)) if 0 < T[i].sum() < T.shape[1]])
        bs = np.array([roc_auc_score(T[i], base[i]) for i in range(len(te)) if 0 < T[i].sum() < T.shape[1]])
        out["per_sample_auroc_mean"] = float(ps.mean())
        out["per_sample_auroc_ci95"] = boot_ci(lambda i: float(ps[i].mean()), len(ps))
        out["per_sample_auroc_baseline_train_mean"] = float(bs.mean())
    else:
        Pc, Tc = P - P.mean(0), T - T.mean(0)
        den = np.sqrt((Pc ** 2).sum(0) * (Tc ** 2).sum(0))
        rf = np.where(den > 0, (Pc * Tc).sum(0) / np.maximum(den, 1e-12), 0)
        out["per_feature_pearson_mean"] = float(rf.mean())
        out["per_feature_pearson_median"] = float(np.median(rf))
        out["fraction_features_r_gt_0.3"] = float((rf > 0.3).mean())
        out["per_feature_pearson_baseline"] = 0.0
        rs = per_sample_r(P, T)
        out["per_sample_pearson_mean"] = float(rs.mean())
        out["per_sample_pearson_ci95"] = boot_ci(lambda i: float(rs[i].mean()), len(rs))
        out["per_sample_pearson_baseline_train_mean"] = float(per_sample_r(base, T).mean())
        out["mse"] = float(((P - T) ** 2).mean())
        out["mse_baseline_train_mean"] = float(((base - T) ** 2).mean())
        out["r2_vs_baseline"] = float(1 - out["mse"] / max(out["mse_baseline_train_mean"], 1e-12))
    return out


# ----------------------------------------------------------------------------- plots
def plot_embedding(Z, codes, classes, te, path, title, pairs=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    sub = te if len(te) <= 3000 else np.sort(RNG.choice(te, 3000, replace=False))
    ncol = 2 if pairs is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.6 * ncol, 4.6))
    axes = np.atleast_1d(axes)
    E = TSNE(n_components=2, init="pca", random_state=0, perplexity=min(30, len(sub) // 4)).fit_transform(Z[sub])
    cmap = plt.get_cmap("tab20")
    if codes is not None:
        for k, c in enumerate(classes):
            m = codes[sub] == k
            if m.any():
                axes[0].scatter(E[m, 0], E[m, 1], s=4, color=cmap(k % 20), label=f"{c} ({m.sum()})")
        axes[0].legend(fontsize=6, markerscale=3, loc="best", ncol=1 + len(classes) // 12)
    else:
        axes[0].scatter(E[:, 0], E[:, 1], s=4)
    axes[0].set_title(f"{title}: joint embedding, test (t-SNE)", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    if pairs is not None:
        (na, Za), (nb, Zb) = pairs
        s2 = sub[: min(len(sub), 1500)]
        both = np.vstack([l2n(Za[s2]), l2n(Zb[s2])])
        E2 = TSNE(n_components=2, init="pca", random_state=0, perplexity=min(30, len(s2) // 4)).fit_transform(both)
        n = len(s2)
        axes[1].scatter(E2[:n, 0], E2[:n, 1], s=4, c="C0", label=na)
        axes[1].scatter(E2[n:, 0], E2[n:, 1], s=4, c="C1", label=nb)
        for i in range(0, n, max(1, n // 40)):
            axes[1].plot([E2[i, 0], E2[n + i, 0]], [E2[i, 1], E2[n + i, 1]], c="grey", lw=0.4, alpha=0.6)
        axes[1].legend(fontsize=7, markerscale=3)
        axes[1].set_title("both modalities in the shared space (grey: same sample)", fontsize=9)
        axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tasks", default="structure,retrieval,collapse")
    ap.add_argument("--labels", default=None, help="label columns for structure (default: all)")
    ap.add_argument("--cross", default=None, help="src->tgt:spec[,src->tgt:spec] (target features from preprocess.py)")
    ap.add_argument("--features-dir", default=None, help="folder holding features/ (default: parent of --run)")
    ap.add_argument("--retrieval-n", type=int, default=2000, help="test pairs used as the retrieval pool")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()

    ds = Dataset(a.data)
    run = Path(a.run)
    cfg = load_json(run / "run_config.json")
    emb = load_embeddings(run)
    tr, va = ds.idx("train"), ds.idx("val")
    te = ds.idx(a.split)  # the samples being scored
    if a.split == "val":  # ridge alpha must then be chosen inside train, not on val
        rng = np.random.default_rng(1)
        perm = rng.permutation(tr)
        cut = int(0.8 * len(perm))
        tr_fit, va_sel = np.sort(perm[:cut]), np.sort(perm[cut:])
    else:
        tr_fit, va_sel = tr, va
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    mod_keys = [k for k in emb if k != "z_joint"]
    metrics = {"run": str(run), "method": cfg.get("method"), "inputs": cfg.get("inputs"),
               "uses_labels": cfg.get("uses_labels", False), "split_evaluated": a.split, "n_evaluated": int(len(te))}

    if "structure" in tasks:
        cols = a.labels.split(",") if a.labels else ds.label_columns
        metrics["structure"] = {}
        for col in cols:
            codes, classes = ds.labels(col)
            if codes is None:
                continue
            metrics["structure"][col] = {k: structure(Z, codes, classes, tr, te) for k, Z in emb.items()}
    if "retrieval" in tasks:
        if len(mod_keys) == 2 and emb[mod_keys[0]].shape[1] == emb[mod_keys[1]].shape[1]:
            r, _ = retrieval(emb[mod_keys[0]], emb[mod_keys[1]], te, a.retrieval_n)
            r["a"], r["b"] = mod_keys[0][2:], mod_keys[1][2:]
            metrics["retrieval"] = r
        else:
            metrics["retrieval"] = {"skipped": "needs two modality embeddings of equal dimension "
                                               f"(have {[(k, emb[k].shape[1]) for k in mod_keys]})"}
    if "collapse" in tasks:
        c = {k: collapse(Z[te]) for k, Z in emb.items()}
        for k, Z in emb.items():
            c[k]["uniformity"] = uniformity(Z[te])
        if len(mod_keys) == 2 and emb[mod_keys[0]].shape[1] == emb[mod_keys[1]].shape[1]:
            A, B = l2n(emb[mod_keys[0]][te]), l2n(emb[mod_keys[1]][te])
            c["alignment_paired_sq_dist"] = float(((A - B) ** 2).sum(1).mean())
            c["alignment_random_pairs_sq_dist"] = float(((A - B[RNG.permutation(len(B))]) ** 2).sum(1).mean())
            c["modality_gap_centroid_dist"] = float(np.linalg.norm(A.mean(0) - B.mean(0)))
        metrics["collapse"] = c
    if "cross" in tasks:
        if not a.cross:
            raise SystemExit("--tasks cross needs --cross src->tgt:spec")
        fdir = Path(a.features_dir) if a.features_dir else run.parent
        metrics["cross"] = {}
        for item in a.cross.split(","):
            pair, spec = item.split(":", 1)
            src, tgt = [s.strip() for s in pair.split("->")]
            key = f"z_{src}"
            if key not in emb:
                metrics["cross"][item] = {"skipped": f"run has no embedding of {src} (keys {list(emb)})"}
                continue
            Y, info = load_features(feature_path(fdir, tgt, spec.strip()))
            binary = set(np.unique(Y[tr][:200])) <= {0.0, 1.0}
            metrics["cross"][item] = cross_predict(emb[key], Y, tr_fit, va_sel, te, binary) | {"target_binary": binary}

    fname = "metrics.json" if a.split == "test" else "metrics_val.json"
    save_json(rounded(metrics), run / fname)
    if not a.no_plot and a.split == "test":
        codes, classes = ds.labels()
        pairs = None
        if len(mod_keys) == 2 and emb[mod_keys[0]].shape[1] == emb[mod_keys[1]].shape[1]:
            pairs = ((mod_keys[0][2:], emb[mod_keys[0]]), (mod_keys[1][2:], emb[mod_keys[1]]))
        plot_embedding(emb["z_joint"], codes, classes, te, run / "embedding.png", cfg.get("method", ""), pairs)

    # console summary
    print(f"evaluated {run} on {a.split} (n={len(te)})")
    for col, d in metrics.get("structure", {}).items():
        for k, s in d.items():
            print(f"  [{col}] {k}: probe bal.acc {s['probe_balanced_accuracy']:.3f} "
                  f"(chance {s['chance_balanced_accuracy']:.3f}), kNN {s['knn15_balanced_accuracy']:.3f}, "
                  f"ARI {s['kmeans_ari']:.3f}")
    r = metrics.get("retrieval", {})
    if "a_to_b" in r:
        print(f"  retrieval {r['a']}->{r['b']}: R@1 {r['a_to_b']['recall_at_1']:.3f} R@5 {r['a_to_b']['recall_at_5']:.3f} "
              f"FOSCTTM {r['a_to_b']['foscttm']:.3f} | {r['b']}->{r['a']}: R@5 {r['b_to_a']['recall_at_5']:.3f} "
              f"(chance R@5 {r['chance']['recall_at_5']:.4f})")
    for item, c in metrics.get("cross", {}).items():
        if "per_sample_pearson_mean" in c:
            print(f"  cross {item}: per-sample r {c['per_sample_pearson_mean']:.3f} "
                  f"(train-mean baseline {c['per_sample_pearson_baseline_train_mean']:.3f}), "
                  f"per-feature r {c['per_feature_pearson_mean']:.3f}")
        elif "per_feature_auroc_mean" in c:
            print(f"  cross {item}: per-feature AUROC {c['per_feature_auroc_mean']:.3f} (0.5), per-sample AUROC "
                  f"{c['per_sample_auroc_mean']:.3f} (train-mean baseline {c['per_sample_auroc_baseline_train_mean']:.3f})")
        else:
            print(f"  cross {item}: {c}")
    print(f"wrote {run / fname}")


if __name__ == "__main__":
    main()
