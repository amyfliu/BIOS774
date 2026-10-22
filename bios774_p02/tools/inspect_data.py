"""inspect_data.py — profile a prepared dataset (facts only, no recommendations).

  python tools/inspect_data.py --data data/pbmc3k --out outputs/r1_pbmc3k

Writes profile.json (+ profile.png: class counts, and an image grid for image
data) into --out. All modeling decisions based on these facts belong to the
agent (see .agents/skills/).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ds = Dataset(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X = ds.X
    tr = ds.split == "train"
    Xs = X[np.random.default_rng(0).choice(len(X), min(len(X), 5000), replace=False)].astype(np.float64)
    n, pdim = X.shape
    nan = int(np.isnan(X).sum()) if np.issubdtype(X.dtype, np.floating) else 0
    is_int = bool(np.allclose(Xs, np.round(Xs)))
    zeros = float((Xs == 0).mean())
    var = Xs.var(0)
    var_sorted = np.sort(var)[::-1]
    prof = {
        "dataset": ds.name, "modality": ds.modality, "n_samples": int(n), "n_features": int(pdim),
        "image_shape_HWC": list(ds.image_shape) if ds.image_shape else None,
        "dtype": str(X.dtype), "memory_MB_float32": round(n * pdim * 4 / 1e6, 1),
        "splits": {s: int((ds.split == s).sum()) for s in np.unique(ds.split)},
        "missing_values": nan,
        "value_range": [float(Xs.min()), float(Xs.max())],
        "integer_valued": is_int, "fraction_zeros": round(zeros, 4),
        "n_features_all_zero_in_sample": int((var == 0).sum()),
        "variance_share_top1pct_features": round(float(var_sorted[: max(1, pdim // 100)].sum() / max(var.sum(), 1e-12)), 4),
        "feature_mean_range": [float(Xs.mean(0).min()), float(Xs.mean(0).max())],
        "feature_sd_ratio_p99_p1": round(float(np.percentile(np.sqrt(var + 1e-12), 99) /
                                               max(np.percentile(np.sqrt(var + 1e-12), 1), 1e-12)), 2),
        "samples_per_feature": round(n / pdim, 3),
    }
    if ds.modality == "tabular":
        lib = Xs.sum(1)
        prof["row_sum"] = {"min": float(lib.min()), "median": float(np.median(lib)), "max": float(lib.max()),
                           "max_over_min": round(float(lib.max() / max(lib.min(), 1e-12)), 2)}
        prof["skewness_median_feature"] = float(np.median(
            ((Xs - Xs.mean(0)) ** 3).mean(0) / np.maximum(Xs.std(0) ** 3, 1e-12)))
        for N in (500, 1000, 2000, 5000):
            if N < pdim:
                prof[f"variance_share_top{N}_features"] = round(float(var_sorted[:N].sum() / var.sum()), 4)
        lx = np.log1p(np.maximum(Xs, 0))
        lv = np.sort(lx.var(0))[::-1]
        prof["after_log1p_variance_share_top2000"] = round(float(lv[:2000].sum() / lv.sum()), 4) if pdim > 2000 else 1.0
    else:
        H, W, C = ds.image_shape
        imgs = Xs.reshape(-1, H, W, C)
        prof["channel_means"] = [round(float(imgs[..., c].mean()), 2) for c in range(C)]
        prof["channel_sds"] = [round(float(imgs[..., c].std()), 2) for c in range(C)]
        dx = np.abs(np.diff(imgs, axis=2)).mean()
        prof["mean_abs_neighbour_pixel_diff"] = round(float(dx), 2)
        prof["neighbour_pixel_correlation"] = round(float(np.corrcoef(imgs[:, :, :-1].ravel(), imgs[:, :, 1:].ravel())[0, 1]), 4)

    if ds.y is not None:
        cnt = {ds.class_name(k): int((ds.y == k).sum()) for k in range(ds.n_classes)}
        cnt_tr = {ds.class_name(k): int(((ds.y == k) & tr).sum()) for k in range(ds.n_classes)}
        v = np.array(list(cnt.values()))
        prof["labels"] = {
            "n_classes": ds.n_classes, "class_counts": cnt, "class_counts_train": cnt_tr,
            "imbalance_ratio_max_over_min": round(float(v.max() / max(v.min(), 1)), 2),
            "smallest_class": min(cnt, key=cnt.get), "smallest_class_train_n": int(min(cnt_tr.values())),
            "normalized_entropy": round(float(-(v / v.sum() * np.log(v / v.sum())).sum() / np.log(len(v))), 4),
        }
        if ds.modality == "image":
            H, W, C = ds.image_shape
            m = {}
            for k in range(ds.n_classes):
                ik = np.where(ds.y == k)[0][:500]
                m[ds.class_name(k)] = round(float(X[ik].astype(np.float64).mean()), 1)
            prof["labels"]["mean_intensity_by_class"] = m
    write_json(out / "profile.json", prof)

    # figure
    ncol = 2 if ds.modality == "image" else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 3.4), squeeze=False)
    if ds.y is not None:
        c = prof["labels"]["class_counts"]
        axes[0, 0].barh(list(c)[::-1], list(c.values())[::-1], color="#4C78A8")
        axes[0, 0].set_xlabel("samples")
        axes[0, 0].tick_params(labelsize=7)
        axes[0, 0].set_title("Samples per class (all splits)")
    if ds.modality == "image":
        H, W, C = ds.image_shape
        rows = []
        for k in range(ds.n_classes):
            ik = np.where(ds.y == k)[0][:8]
            rows.append(np.concatenate([X[i].reshape(H, W, C) for i in ik], 1))
        axes[0, 1].imshow(np.concatenate(rows, 0).astype(np.uint8).squeeze())
        axes[0, 1].set_yticks([H * k + H / 2 for k in range(ds.n_classes)])
        axes[0, 1].set_yticklabels([ds.class_name(k)[:18] for k in range(ds.n_classes)], fontsize=6)
        axes[0, 1].set_xticks([])
        axes[0, 1].set_title("Real examples, one row per class")
    fig.tight_layout()
    fig.savefig(out / "profile.png", dpi=110)
    print(f"[inspect] wrote {out / 'profile.json'}")


if __name__ == "__main__":
    main()
