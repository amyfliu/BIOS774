"""evaluate.py — evaluate generated data (and/or likelihood scores) for a run.

  python tools/evaluate.py --run outputs/r2/ddpm --tasks fidelity,memorization,utility --utility tstr
  python tools/evaluate.py --run outputs/r1/cvae --tasks fidelity,memorization,utility \
         --utility augment --samples-tag balance
  python tools/evaluate.py --run outputs/r3/flow --tasks anomaly --anomaly-classes background,debris

Every metric is reported next to a REAL-vs-REAL reference value computed the
same way (e.g. real-train vs real-test), because absolute numbers mean little
on their own. The script computes; it does not judge — interpretation
thresholds live in .agents/skills/evaluation/SKILL.md.

Tasks
  fidelity      Frechet distance, k-NN precision/recall (Kynkaanniemi et al. 2019),
                RBF-MMD, classifier two-sample test (C2ST), label consistency
                (does a real-trained classifier agree with the synthetic labels?),
                per-class precision/recall, 2-D PCA picture of real vs synthetic.
  memorization  distance from each synthetic sample to its nearest TRAIN sample,
                compared with the same distance for held-out real TEST samples.
  utility       --utility tstr    : train-on-synthetic / test-on-real vs train-on-real
                --utility augment : real train + synthetic vs real only, class
                                    re-weighting and random oversampling baselines
  anomaly       AUROC / AUPRC of -log p from score.py (needs scores_test.npz), a PCA
                reconstruction-error baseline, and (images) a check of whether the
                likelihood just tracks image complexity.

Feature space used for comparisons
  tabular: model space (after the run's preprocessing) projected to <=50 PCs fit on real train
  image  : 128-d penultimate features of a small CNN trained once on real train
           (cached in outputs/_cache/<dataset>/), a domain-specific stand-in for the
           Inception network used by FID.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported (Apple GPU)

import argparse
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy import linalg, stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.metrics.pairwise import euclidean_distances  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, Prep, get_device, read_json, set_seed, stratified_subsample, write_json  # noqa: E402

import warnings  # noqa: E402

# C2ST's small MLP may hit max_iter when real and synthetic are trivially separable;
# its accuracy is still the quantity we want.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).resolve().parent.parent
MAX_EVAL = 3000


# ============================================================================ image classifier
class GlobalAvgPool(nn.Module):
    """mean over H, W — avoids AdaptiveAvgPool2d + Flatten (view), which fails in MPS backward."""

    def forward(self, x):
        return x.mean(dim=(2, 3))


class SmallCNN(nn.Module):
    def __init__(self, C, n_classes):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(C, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            GlobalAvgPool(), nn.Identity(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.f(x))


def _img_tensor(U, shape, dev):
    H, W, C = shape
    # ascontiguousarray: MPS backward fails on the strided (transposed) layout.
    a = np.ascontiguousarray(U.reshape(-1, H, W, C).transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
    return torch.from_numpy(a).to(dev).contiguous()


def train_cnn(U, y, shape, n_classes, dev, epochs=5, weights=None, seed=0):
    set_seed(seed)
    net = SmallCNN(shape[2], n_classes).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    X = _img_tensor(U, shape, dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    w = torch.tensor(weights, dtype=torch.float32, device=dev) if weights is not None else None
    lossf = nn.CrossEntropyLoss(weight=w)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        for b in range(0, len(X), 128):
            bi = perm[b:b + 128]
            if len(bi) < 2:
                continue
            loss = lossf(net(X[bi]), yt[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


@torch.no_grad()
def cnn_apply(net, U, shape, dev, what="pred"):
    out = []
    for b in range(0, len(U), 512):
        x = _img_tensor(U[b:b + 512], shape, dev)
        out.append((net.f(x) if what == "feat" else net(x).argmax(1)).cpu().numpy())
    return np.concatenate(out)


def eval_cnn(ds, dev):
    """Train once on real train (<=20k images) and cache; used for features + label checks."""
    cache = ROOT / "outputs" / "_cache" / ds.name
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / "eval_cnn.pt"
    net = SmallCNN(ds.image_shape[2], ds.n_classes).to(dev)
    if f.exists():
        net.load_state_dict(torch.load(f, map_location=dev))
        return net.eval(), read_json(cache / "eval_cnn.json")
    tr = ds.indices("train", max_n=20000)
    te = ds.indices("test")
    net = train_cnn(ds.X[tr].astype(np.uint8), ds.y[tr], ds.image_shape, ds.n_classes, dev, epochs=4)
    acc = float((cnn_apply(net, ds.X[te].astype(np.uint8), ds.image_shape, dev) == ds.y[te]).mean())
    torch.save(net.state_dict(), f)
    info = {"n_train": int(len(tr)), "real_test_accuracy": round(acc, 4)}
    write_json(cache / "eval_cnn.json", info)
    return net, info


# ============================================================================ metrics
def frechet(A, B):
    mu1, mu2 = A.mean(0), B.mean(0)
    s1, s2 = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = (covmean[0] if isinstance(covmean, tuple) else covmean).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def prec_recall(real, fake, k=5):
    k = max(1, min(k, len(real) - 1, len(fake) - 1))
    rr = NearestNeighbors(n_neighbors=k + 1).fit(real).kneighbors(real)[0][:, -1]
    fr = NearestNeighbors(n_neighbors=k + 1).fit(fake).kneighbors(fake)[0][:, -1]
    d_fr = euclidean_distances(fake, real) if len(real) * len(fake) <= 1e7 else None
    if d_fr is None:
        nn_r = NearestNeighbors(n_neighbors=min(50, len(real))).fit(real)
        dist, ind = nn_r.kneighbors(fake)
        precision = float((dist <= rr[ind]).any(1).mean())
        nn_f = NearestNeighbors(n_neighbors=min(50, len(fake))).fit(fake)
        dist, ind = nn_f.kneighbors(real)
        recall = float((dist <= fr[ind]).any(1).mean())
    else:
        precision = float((d_fr <= rr[None]).any(1).mean())
        recall = float((d_fr.T <= fr[None]).any(1).mean())
    return precision, recall


def mmd_rbf(A, B, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    Z = np.vstack([A, B])
    d2 = euclidean_distances(Z, squared=True)
    gamma = 1.0 / max(np.median(d2[d2 > 0]), 1e-8)
    K = np.exp(-gamma * d2)
    m = len(A)
    kxx, kyy, kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (len(B) * (len(B) - 1)) - 2 * kxy.mean())


def c2st(A, B, seed=0):
    n = min(len(A), len(B), 2000)
    rng = np.random.default_rng(seed)
    X = np.vstack([A[rng.choice(len(A), n, replace=False)], B[rng.choice(len(B), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    X = StandardScaler().fit_transform(X)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean())


def sub(X, n, seed=0, y=None):
    if len(X) <= n:
        return (X, y) if y is not None else X
    i = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return (X[i], y[i]) if y is not None else X[i]


def nn_dist(query, ref):
    return NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(query)[0][:, 0]


# ============================================================================ classifiers for utility
def fit_predict(ctx, Xtr, ytr, Xte, weights=None, seed=0):
    """Tabular: logistic regression on eval features. Image: small CNN on pixels."""
    if ctx["modality"] == "image":
        net = train_cnn(Xtr, ytr, ctx["shape"], ctx["K"], ctx["dev"], epochs=ctx["clf_epochs"],
                        weights=weights, seed=seed)
        return cnn_apply(net, Xte, ctx["shape"], ctx["dev"])
    sc = StandardScaler().fit(Xtr)
    cw = None if weights is None else {k: float(w) for k, w in enumerate(weights)}
    clf = LogisticRegression(max_iter=3000, C=0.5, class_weight=cw).fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def scores(y, p, K, names, focus=None):
    rec = recall_score(y, p, labels=range(K), average=None, zero_division=0)
    f1 = f1_score(y, p, labels=range(K), average=None, zero_division=0)
    out = {"accuracy": float((y == p).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, p)),
           "macro_f1": float(f1.mean()),
           "per_class_recall": {names[k]: round(float(rec[k]), 4) for k in range(K)},
           "per_class_f1": {names[k]: round(float(f1[k]), 4) for k in range(K)}}
    if focus:
        out["focus_classes_mean_f1"] = float(np.mean([f1[k] for k in focus]))
        out["focus_classes_mean_recall"] = float(np.mean([rec[k] for k in focus]))
    return out


def paired_bootstrap(y, p_a, p_b, K, n_boot=1000, seed=0):
    """95% CI for macro-F1(b) - macro-F1(a) by resampling test rows."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], p_b[i], labels=range(K), average="macro", zero_division=0)
                 - f1_score(y[i], p_a[i], labels=range(K), average="macro", zero_division=0))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


# ============================================================================ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--tasks", default="fidelity,memorization")
    p.add_argument("--samples-tag", default="")
    p.add_argument("--utility", choices=["tstr", "augment"], default=None)
    p.add_argument("--real-max", type=int, default=None,
                   help="augment: use only this many real training rows (simulates scarce data)")
    p.add_argument("--anomaly-classes", default=None)
    p.add_argument("--clf-epochs", type=int, default=5)
    p.add_argument("--data", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    set_seed(a.seed)
    run = Path(a.run)
    cfg = read_json(run / "run_config.json")
    ds = Dataset(a.data or ROOT / "data" / cfg["dataset"])
    prep = Prep.load(run / "prep.pkl")
    dev = get_device(a.device)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    sfx = f"_{a.samples_tag}" if a.samples_tag else ""
    K, names = ds.n_classes, ds.class_names()
    res = {"run": str(run), "samples": f"samples{sfx}.npz", "dataset": ds.name, "tasks": tasks,
           "feature_space": None}

    inc = ds.resolve_classes(",".join(cfg.get("include_classes", [])))
    exc = ds.resolve_classes(",".join(cfg.get("exclude_classes", [])))
    tr_idx = ds.indices("train", include=inc, exclude=exc)
    te_idx = ds.indices("test", include=inc, exclude=exc)

    need_samples = any(t in tasks for t in ("fidelity", "memorization", "utility"))

    # ---------------------------------------------------------------- feature map
    if ds.modality == "image" and not need_samples:
        feat = raw_real = None          # anomaly-only: no learned feature space needed
    elif ds.modality == "image":
        net, info = eval_cnn(ds, dev)
        res["feature_space"] = {"type": "eval_cnn_penultimate_128d", **info}
        feat = lambda U: cnn_apply(net, U.astype(np.uint8), ds.image_shape, dev, "feat")  # noqa: E731
        raw_real = lambda idx: ds.X[idx].astype(np.uint8)                                   # noqa: E731
    else:
        Mtr = prep.transform(ds.X[tr_idx]).astype(np.float64)   # float64: avoids spurious BLAS warnings on macOS
        k = int(min(50, Mtr.shape[1], len(Mtr) - 1))
        mu = Mtr.mean(0)
        V = np.linalg.svd(Mtr - mu, full_matrices=False)[2][:k]
        feat = lambda M: (M.astype(np.float64) - mu) @ V.T                                                    # noqa: E731
        raw_real = lambda idx: prep.transform(ds.X[idx])                                    # noqa: E731
        res["feature_space"] = {"type": f"model_space_pca{k}", "prep": prep.spec}

    if need_samples:
        S = np.load(run / f"samples{sfx}.npz", allow_pickle=True)
        labelled = bool(S["labelled"])
        ys = S["y"]
        syn_raw = S["X_uint8"] if ds.modality == "image" else S["X"]
        F_syn = feat(syn_raw)
        tr_sub = sub(tr_idx, 20000, 7) if ds.modality == "image" else tr_idx
        F_tr, y_trs = feat(raw_real(tr_sub)), ds.y[tr_sub]
        F_te = feat(raw_real(te_idx))
        y_te = ds.y[te_idx]
        # Class-matched real reference: if only some classes were generated (e.g. class
        # balancing), compare against real data of those classes only.
        syn_classes = np.unique(ys) if labelled else np.arange(max(K, 1))
        m_te = np.isin(y_te, syn_classes) if labelled else np.ones(len(y_te), bool)
        m_tr = np.isin(y_trs, syn_classes) if labelled else np.ones(len(y_trs), bool)
        res["real_reference_classes"] = [names[k] for k in syn_classes] if labelled else "all"

    # ---------------------------------------------------------------- fidelity
    if "fidelity" in tasks:
        A, yA = sub(F_te[m_te], MAX_EVAL, 1, y_te[m_te])
        B, yB = sub(F_syn, MAX_EVAL, 2, ys)
        R = sub(F_tr[m_tr], MAX_EVAL, 3)
        pr, rc = prec_recall(A, B)
        pr0, rc0 = prec_recall(A, R)
        fid = {"frechet": frechet(A, B), "precision": pr, "recall": rc, "mmd_rbf": mmd_rbf(A, B),
               "c2st_accuracy": c2st(A, B),
               "reference_real_train_vs_real_test": {"frechet": frechet(A, R), "precision": pr0, "recall": rc0,
                                                     "mmd_rbf": mmd_rbf(A, R), "c2st_accuracy": c2st(A, R)},
               "n_real": len(A), "n_synthetic": len(B),
               "frechet_unreliable_small_n": bool(min(len(A), len(B)) < 2 * A.shape[1])}
        if labelled:
            if ds.modality == "image":
                pred_syn = cnn_apply(net, syn_raw.astype(np.uint8), ds.image_shape, dev)
                ref_acc = info["real_test_accuracy"]
            else:
                sc = StandardScaler().fit(F_tr)
                clf = LogisticRegression(max_iter=3000, C=0.5).fit(sc.transform(F_tr), ds.y[tr_idx])
                pred_syn = clf.predict(sc.transform(F_syn))
                ref_acc = float((clf.predict(sc.transform(F_te)) == y_te).mean())
            fid["label_consistency"] = {
                "accuracy_on_synthetic": float((pred_syn == ys).mean()),
                "reference_accuracy_on_real_test": ref_acc,
                "per_class": {names[k]: round(float((pred_syn[ys == k] == k).mean()), 4)
                              for k in np.unique(ys)}}
            per = {}
            F_all_real = np.vstack([F_tr, F_te]) if ds.modality != "image" else None
            y_all_real = np.r_[ds.y[tr_idx], y_te] if ds.modality != "image" else None
            for kk in np.unique(ys):
                real_k = F_te[y_te == kk]
                ref = "test"
                if len(real_k) < 20 and F_all_real is not None:
                    real_k, ref = F_all_real[y_all_real == kk], "train+test (few test cells)"
                syn_k = F_syn[ys == kk]
                if len(real_k) >= 3 and len(syn_k) >= 3:
                    pk, rk = prec_recall(sub(real_k, 1500), sub(syn_k, 1500))
                    per[names[kk]] = {"precision": round(pk, 4), "recall": round(rk, 4),
                                      "n_real_ref": int(len(real_k)), "reference": ref}
            fid["per_class"] = per
        res["fidelity"] = fid
        # picture: shared 2-D PCA of real test vs synthetic
        P = np.vstack([A, B])
        mu2 = P.mean(0)
        V2 = np.linalg.svd(P - mu2, full_matrices=False)[2][:2]
        Ra, Sb = (A - mu2) @ V2.T, (B - mu2) @ V2.T
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
        cmap = plt.get_cmap("tab10")
        for axi, Z, yy, title in [(ax[0], Ra, yA, "real (test)"), (ax[1], Sb, yB, "synthetic")]:
            for kk in np.unique(np.r_[yA, yB]):
                m = yy == kk
                if labelled or title.startswith("real"):
                    axi.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.5, color=cmap(kk % 10), label=names[kk][:20])
                else:
                    axi.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.5, color="gray")
                    break
            axi.set_title(title)
        ax[0].legend(fontsize=6, markerscale=3, loc="best")
        fig.suptitle(f"{cfg['model']} — shared PCA of evaluation features", fontsize=10)
        fig.tight_layout()
        fig.savefig(run / f"embedding{sfx}.png", dpi=110)
        plt.close(fig)

    # ---------------------------------------------------------------- memorization
    if "memorization" in tasks:
        tr_pool = tr_idx[np.isin(ds.y[tr_idx], syn_classes)] if labelled else tr_idx
        te_pool = te_idx[np.isin(ds.y[te_idx], syn_classes)] if labelled else te_idx
        te_pool = sub(te_pool, MAX_EVAL, 5)
        if ds.modality == "image":
            space = "pixels_scaled_0_1"
            vec = lambda idx: ds.X[idx].astype(np.float32) / 255           # noqa: E731
            syn_s = sub(syn_raw, MAX_EVAL, 4).astype(np.float32) / 255
            tr_pool = sub(tr_pool, 30000, 6)
        else:
            space = f"model_space ({prep.spec})"
            vec = raw_real                                                 # noqa: E731
            syn_s = sub(syn_raw, MAX_EVAL, 4)
        T_all, T_te = vec(tr_pool), vec(te_pool)
        d_syn, d_ref = nn_dist(syn_s, T_all), nn_dist(T_te, T_all)
        # Membership test: pool an equal-size, class-stratified train subset with the test
        # set; a non-memorizing generator's samples should land nearest to each about equally.
        tr_eq = stratified_subsample(tr_pool, ds.y, len(te_pool), 8) if len(tr_pool) > len(te_pool) else tr_pool
        pooled = np.vstack([vec(tr_eq), T_te])
        nn_i = NearestNeighbors(n_neighbors=1).fit(pooled).kneighbors(syn_s)[1][:, 0]
        p1, p5 = np.percentile(d_ref, 1), np.percentile(d_ref, 5)
        res["memorization"] = {
            "space": space,
            "frac_nearest_real_is_train": float((nn_i < len(tr_eq)).mean()),
            "frac_nearest_real_is_train_expected_if_no_memorization": round(len(tr_eq) / len(pooled), 3),
            "n_train_in_pool": int(len(tr_eq)), "n_test_in_pool": int(len(te_pool)),
            "median_nn_dist_synthetic_to_train": float(np.median(d_syn)),
            "median_nn_dist_real_test_to_train": float(np.median(d_ref)),
            "ratio_synthetic_over_real": float(np.median(d_syn) / max(np.median(d_ref), 1e-12)),
            "frac_synthetic_closer_than_real_p1": float((d_syn < p1).mean()),
            "frac_synthetic_closer_than_real_p5": float((d_syn < p5).mean()),
            "n_exact_copies": int((d_syn < 1e-6).sum()),
        }
        fig, ax = plt.subplots(figsize=(5, 3))
        bins = np.linspace(0, np.percentile(np.r_[d_syn, d_ref], 99), 50)
        ax.hist(d_ref, bins, alpha=0.6, density=True, label="real test → train")
        ax.hist(d_syn, bins, alpha=0.6, density=True, label="synthetic → train")
        ax.set_xlabel("distance to nearest training sample")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run / f"memorization{sfx}.png", dpi=110)
        plt.close(fig)

    # ---------------------------------------------------------------- utility
    if "utility" in tasks:
        if not labelled:
            sys.exit("utility needs labelled (class-conditional) samples")
        ctx = {"modality": ds.modality, "shape": ds.image_shape, "K": K, "dev": dev, "clf_epochs": a.clf_epochs}
        if ds.modality == "image":
            Xte_c = ds.X[te_idx].astype(np.uint8)
            real_pool, syn_c = tr_idx, syn_raw.astype(np.uint8)
            get_real = lambda idx: ds.X[idx].astype(np.uint8)  # noqa: E731
        else:
            Xte_c, syn_c = F_te, F_syn
            get_real = lambda idx: feat(raw_real(idx))  # noqa: E731
        util = {"mode": a.utility, "classifier": "small CNN" if ds.modality == "image" else "logistic regression"}
        if a.utility == "tstr":
            # real reference with the SAME per-class counts as the synthetic set
            rng = np.random.default_rng(a.seed)
            ref = np.concatenate([rng.choice(real_pool[ds.y[real_pool] == k], min((ys == k).sum(),
                                  (ds.y[real_pool] == k).sum()), replace=False) for k in np.unique(ys)])
            p_syn = fit_predict(ctx, syn_c, ys, Xte_c)
            p_real = fit_predict(ctx, get_real(ref), ds.y[ref], Xte_c)
            util["train_on_synthetic"] = scores(y_te, p_syn, K, names)
            util["train_on_real_same_size"] = scores(y_te, p_real, K, names)
            util["n_train_each"] = int(len(ys))
            util["tstr_over_trtr_accuracy"] = util["train_on_synthetic"]["accuracy"] / max(
                util["train_on_real_same_size"]["accuracy"], 1e-8)
            util["macro_f1_diff_95ci_(syn_minus_real)"] = paired_bootstrap(y_te, p_real, p_syn, K)
        else:
            base = stratified_subsample(tr_idx, ds.y, a.real_max, a.seed) if a.real_max else tr_idx
            yb = ds.y[base]
            focus = [int(k) for k in np.unique(ys)]
            Xb = get_real(base)
            counts = np.bincount(yb, minlength=K).astype(float)
            w = counts.sum() / (K * np.maximum(counts, 1))
            rng = np.random.default_rng(a.seed)
            over = np.concatenate([rng.choice(np.where(yb == k)[0], (ys == k).sum(), replace=True) for k in focus])
            arms = {
                "real_only": (Xb, yb, None),
                "real_class_weighted": (Xb, yb, w),
                "real_random_oversample": (np.concatenate([Xb, Xb[over]]), np.r_[yb, yb[over]], None),
                "real_plus_synthetic": (np.concatenate([Xb, syn_c]), np.r_[yb, ys], None),
            }
            preds = {}
            for name, (Xa, ya, wa) in arms.items():
                preds[name] = fit_predict(ctx, Xa, ya, Xte_c, weights=wa, seed=a.seed)
                util[name] = scores(y_te, preds[name], K, names, focus)
            util["n_real_train"] = int(len(base))
            util["n_synthetic_added"] = {names[k]: int((ys == k).sum()) for k in focus}
            util["focus_classes"] = [names[k] for k in focus]
            util["macro_f1_diff_95ci"] = {
                "synthetic_minus_real_only": paired_bootstrap(y_te, preds["real_only"], preds["real_plus_synthetic"], K),
                "synthetic_minus_class_weighted": paired_bootstrap(y_te, preds["real_class_weighted"],
                                                                   preds["real_plus_synthetic"], K),
                "synthetic_minus_oversample": paired_bootstrap(y_te, preds["real_random_oversample"],
                                                               preds["real_plus_synthetic"], K)}
            util["n_test_per_focus_class"] = {names[k]: int((y_te == k).sum()) for k in focus}
        res["utility"] = util

    # ---------------------------------------------------------------- anomaly
    if "anomaly" in tasks:
        sf = run / "scores_test.npz"
        if not sf.exists():
            sys.exit("anomaly task needs scores_test.npz — run tools/score.py --run ... --split test first")
        Sc = np.load(sf)
        anom = ds.resolve_classes(a.anomaly_classes)
        if not anom:
            sys.exit("--anomaly-classes is required for the anomaly task")
        y_s, lp = Sc["y"].astype(int), Sc["logp_per_dim"]
        is_anom = np.isin(y_s, anom).astype(int)
        score = -lp
        out = {"anomaly_classes": [names[k] for k in anom],
               "trained_on_classes": cfg.get("include_classes") or [n for n in names if n not in cfg.get("exclude_classes", [])],
               "n_test": int(len(y_s)), "n_anomalous": int(is_anom.sum()),
               "auroc": float(roc_auc_score(is_anom, score)),
               "auprc": float(average_precision_score(is_anom, score)),
               "auprc_chance": float(is_anom.mean()),
               "mean_logp_per_dim_by_class": {names[k]: round(float(lp[y_s == k].mean()), 4) for k in np.unique(y_s)}}
        # baseline: PCA reconstruction error, fit on the same training classes
        Mtr = prep.transform(ds.X[sub(tr_idx, 20000)]).reshape(min(len(tr_idx), 20000), -1).astype(np.float64)
        Mte = prep.transform(ds.X[Sc["idx"]]).reshape(len(Sc["idx"]), -1).astype(np.float64)
        k = int(min(50, Mtr.shape[1] - 1))
        mu = Mtr.mean(0)
        V = np.linalg.svd(Mtr - mu, full_matrices=False)[2][:k]
        err = (((Mte - mu) - ((Mte - mu) @ V.T) @ V) ** 2).mean(1)
        out["baseline_pca_recon_error_auroc"] = float(roc_auc_score(is_anom, err))
        if ds.modality == "image":
            comp = np.array([len(zlib.compress(ds.X[i].astype(np.uint8).tobytes(), 9)) for i in Sc["idx"]])
            rho = stats.spearmanr(lp, comp).correlation
            # Serra et al. (2020): subtract a generic compressor's code length from the
            # model's code length, so "simple" images are not automatically called typical.
            D = ds.X.shape[1]
            corrected = (-lp / np.log(2)) - comp * 8.0 / D
            out["complexity_check"] = {
                "auroc_complexity_corrected_score": float(roc_auc_score(is_anom, corrected)),
                "spearman_logp_vs_compressed_size": float(rho),
                "auroc_of_complexity_alone": float(roc_auc_score(is_anom, -comp)),
                "auroc_of_complexity_alone_reversed": float(roc_auc_score(is_anom, comp))}
        res["anomaly"] = out
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for kk in np.unique(y_s):
            ax.hist(lp[y_s == kk], 40, histtype="step", density=True,
                    label=names[kk][:20] + (" (anomaly)" if kk in anom else ""),
                    linestyle="--" if kk in anom else "-")
        ax.set_xlabel("log p(x) per dimension (higher = more typical)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(run / "anomaly_scores.png", dpi=110)
        plt.close(fig)

    out_f = run / f"metrics{sfx}.json"
    write_json(out_f, res)
    print(f"[evaluate] wrote {out_f}")
    for t in ("fidelity", "memorization", "utility", "anomaly"):
        if t in res:
            short = {k: v for k, v in res[t].items() if not isinstance(v, (dict, list))}
            print(f"  {t}: {short}")


if __name__ == "__main__":
    main()
