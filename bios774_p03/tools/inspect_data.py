"""inspect_data.py — profile a prepared multimodal dataset (facts only, no decisions).

  python tools/inspect_data.py --data data/pbmc_multiome --out outputs/<run_id>

Writes <out>/profile.json and <out>/profile.png. Statistics that could inform
preprocessing are computed on the train split only.

Per modality kind:
  counts  features, sparsity, library size, how binary the values are,
          mean-variance (overdispersion), features detected in >= 1% of cells
  image   image sizes, colour vs grayscale, intensity
  text    words per text, vocabulary size, most frequent words, how often a
          text literally names its own label, and how well the words ALONE
          predict each label (TF-IDF + logistic regression, train -> val):
          near-perfect means a label task is a text shortcut, not a test of alignment
Plus: split sizes, label balance per label column, pairing description.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, rounded, save_json  # noqa: E402

WORD = re.compile(r"[a-z][a-z0-9\-]+")


def q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, dtype=float)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def profile_counts(ds: Dataset, mod: str, tr: np.ndarray) -> dict:
    X = ds.counts(mod)
    Xt = X[tr]
    n, d = Xt.shape
    lib = np.asarray(Xt.sum(1)).ravel()
    det = np.asarray((Xt > 0).sum(1)).ravel()
    feat_cells = np.asarray((Xt > 0).sum(0)).ravel() / n
    vals = Xt.data
    mean = np.asarray(Xt.mean(0)).ravel()
    sq = np.asarray(Xt.multiply(Xt).mean(0)).ravel()
    var = sq - mean ** 2
    top = np.argsort(-mean)[: min(2000, d)]
    ok = mean[top] > 0
    slope = float(np.polyfit(np.log10(mean[top][ok]), np.log10(np.maximum(var[top][ok], 1e-12)), 1)[0]) if ok.sum() > 10 else None
    disp = float(np.median((var[top][ok] - mean[top][ok]) / mean[top][ok] ** 2)) if ok.sum() > 10 else None
    return {
        "kind": "counts",
        "n_features": int(d),
        "fraction_zero": 1 - Xt.nnz / (n * d),
        "library_size": q(lib),
        "features_detected_per_sample": q(det),
        "fraction_nonzero_values_equal_1": float((vals == 1).mean()) if len(vals) else None,
        "fraction_nonzero_values_le_2": float((vals <= 2).mean()) if len(vals) else None,
        "max_value": float(vals.max()) if len(vals) else 0.0,
        "features_in_ge_1pct_samples": int((feat_cells >= 0.01).sum()),
        "features_in_ge_10pct_samples": int((feat_cells >= 0.10).sum()),
        "mean_variance_loglog_slope_top2000": slope,
        "median_overdispersion_top2000": disp,
        "_plot": {"lib": lib, "det": det},
    }


def profile_image(ds: Dataset, mod: str, tr: np.ndarray, n_check=300, seed=0) -> dict:
    from PIL import Image

    paths = ds.image_paths(mod)
    rng = np.random.default_rng(seed)
    pick = rng.choice(tr, size=min(n_check, len(tr)), replace=False)
    w, h, gray, mean_int = [], [], [], []
    for i in pick:
        im = Image.open(paths[i]).convert("RGB")
        a = np.asarray(im, dtype=np.float32)
        w.append(im.size[0]); h.append(im.size[1])
        gray.append(float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 2.0))
        mean_int.append(float(a.mean()))
    return {
        "kind": "image",
        "n_images_checked": int(len(pick)),
        "width": q(w), "height": q(h),
        "aspect_ratio_w_over_h": q(np.array(w) / np.array(h)),
        "fraction_grayscale": float(np.mean(gray)),
        "mean_intensity_0_255": q(mean_int),
        "_plot": {"paths": [str(paths[i]) for i in pick[:8]]},
    }


def profile_text(ds: Dataset, mod: str, tr: np.ndarray) -> dict:
    texts = ds.texts(mod)
    toks = [WORD.findall(t.lower()) for t in texts]
    lens = np.array([len(t) for t in toks])
    vocab = Counter(w for i in tr for w in toks[i])
    stop = {"the", "of", "and", "in", "with", "a", "an", "to", "is", "are", "on", "at", "for", "from", "by",
            "as", "was", "were", "or", "this", "that", "which", "be", "after", "showing", "shows"}
    common = [(w, c) for w, c in vocab.most_common(80) if w not in stop][:20]
    out = {
        "kind": "text",
        "words_per_text": q(lens[tr]),
        "fraction_empty": float((lens == 0).mean()),
        "vocabulary_size_train": int(len(vocab)),
        "words_seen_ge_5_times_train": int(sum(c >= 5 for c in vocab.values())),
        "most_common_words_train": common,
        "duplicate_text_fraction": float(1 - len(set(texts)) / max(len(texts), 1)),
        "_plot": {"lens": lens[tr]},
    }
    # label shortcut: does the text contain its own label's name?
    shortcut = {}
    for col in ds.label_columns:
        labs = ds.obs[col].astype(str).values
        rates = {}
        for lab in np.unique(labs[tr]):
            key = [w for w in WORD.findall(lab.lower()) if len(w) >= 2]
            if not key:
                continue
            idx = tr[labs[tr] == lab]
            hit = np.mean([all(any(k in w or w in k for w in toks[i] if len(w) >= 2) for k in key[:1])
                           for i in idx]) if len(idx) else 0.0
            rates[lab] = float(hit)
        if rates:  # literal label-word matches only (misses synonyms; see label_predictable_from_words_alone)
            shortcut[col] = {"per_class": rates, "mean": float(np.mean(list(rates.values())))}
    out["texts_naming_own_label"] = shortcut
    # synonym-proof version: how well do the words alone predict each label?
    # (TF-IDF + logistic regression, fit on train, scored on val)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    va = ds.idx("val")
    pred = {}
    if len(va):
        vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True).fit([texts[i] for i in tr])
        A = vec.transform(texts)
        for col in ds.label_columns:
            labs = ds.obs[col].astype(str).values
            if len(np.unique(labs[tr])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(A[tr], labs[tr])
            pred[col] = {"val_balanced_accuracy": float(balanced_accuracy_score(labs[va], clf.predict(A[va]))),
                         "chance": 1 / len(np.unique(labs[tr]))}
    out["label_predictable_from_words_alone"] = pred
    return out


def profile_labels(ds: Dataset) -> dict:
    out = {}
    for col in ds.label_columns:
        labs = ds.obs[col].astype(str)
        per = {s: labs[ds.split == s].value_counts().to_dict() for s in ("train", "val", "test")}
        tr = per["train"]
        out[col] = {
            "n_classes": int(labs.nunique()),
            "counts_by_split": per,
            "smallest_class_train": int(min(tr.values())) if tr else 0,
            "largest_class_train": int(max(tr.values())) if tr else 0,
            "imbalance_ratio_train": float(max(tr.values()) / max(min(tr.values()), 1)) if tr else None,
            "majority_class_fraction_test": float(max(per["test"].values()) / max(sum(per["test"].values()), 1))
            if per["test"] else None,
        }
    return out


def plot(profile: dict, ds: Dataset, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    mods = profile["modalities"]
    n_panels = 1 + len(mods)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.4))
    axes = np.atleast_1d(axes)
    col = ds.primary_label
    if col:
        vc = ds.obs.loc[ds.split == "train", col].astype(str).value_counts()
        axes[0].barh(vc.index[::-1], vc.values[::-1], color="#4C72B0")
        axes[0].set_title(f"train samples per '{col}'", fontsize=9)
        axes[0].tick_params(labelsize=7)
    for ax, (name, p) in zip(axes[1:], mods.items()):
        pl = p.get("_plot", {})
        if p["kind"] == "counts":
            ax.hist(np.log10(np.maximum(pl["lib"], 1)), bins=50, color="#55A868")
            ax.set_xlabel("log10 total counts per sample", fontsize=8)
            ax.set_title(f"{name}: {p['n_features']} features, {100 * p['fraction_zero']:.1f}% zeros", fontsize=9)
        elif p["kind"] == "text":
            ax.hist(pl["lens"], bins=50, color="#C44E52")
            ax.set_xlabel("words per text", fontsize=8)
            ax.set_title(f"{name}: median {p['words_per_text']['p50']:.0f} words", fontsize=9)
        elif p["kind"] == "image":
            ims = [np.asarray(Image.open(x).convert("RGB").resize((96, 96))) for x in pl["paths"][:8]]
            if ims:
                rows = [np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)]
                rows = [r if r.shape[1] == rows[0].shape[1] else np.pad(r, ((0, 0), (0, rows[0].shape[1] - r.shape[1]), (0, 0)))
                        for r in rows]
                ax.imshow(np.vstack(rows)); ax.axis("off")
            ax.set_title(f"{name}: random train images", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = Dataset(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = ds.idx("train")

    mods = {}
    for name, m in ds.modalities.items():
        fn = {"counts": profile_counts, "image": profile_image, "text": profile_text}[m["kind"]]
        mods[name] = fn(ds, name, tr)
    prof = {
        "dataset": ds.name,
        "description": ds.meta.get("description", ""),
        "pairing": ds.meta.get("pairing", ""),
        "n_samples": len(ds),
        "splits": {s: int((ds.split == s).sum()) for s in ("train", "val", "test")},
        "modalities": mods,
        "labels": profile_labels(ds),
        "primary_label": ds.primary_label,
    }
    plot(prof, ds, out / "profile.png")
    for p in mods.values():
        p.pop("_plot", None)
    save_json(rounded(prof), out / "profile.json")

    # short human-readable summary
    print(f"{ds.name}: {len(ds)} samples {prof['splits']}")
    for name, p in mods.items():
        if p["kind"] == "counts":
            print(f"  {name} (counts): {p['n_features']} features, {100 * p['fraction_zero']:.1f}% zeros, "
                  f"median library {p['library_size']['p50']:.0f}, "
                  f"{100 * p['fraction_nonzero_values_equal_1']:.0f}% of non-zeros are 1")
        elif p["kind"] == "image":
            print(f"  {name} (image): median {p['width']['p50']:.0f}x{p['height']['p50']:.0f}, "
                  f"{100 * p['fraction_grayscale']:.0f}% grayscale")
        else:
            sc = {k: round(v["mean"], 2) for k, v in p["texts_naming_own_label"].items()}
            pr = {k: f"{v['val_balanced_accuracy']:.2f} (chance {v['chance']:.2f})"
                  for k, v in p.get("label_predictable_from_words_alone", {}).items()}
            print(f"  {name} (text): median {p['words_per_text']['p50']:.0f} words, vocab {p['vocabulary_size_train']}, "
                  f"texts naming own label {sc}, label predicted from words alone (val bal.acc) {pr}")
    for col, l in prof["labels"].items():
        print(f"  label {col}: {l['n_classes']} classes, smallest train class {l['smallest_class_train']}, "
              f"majority fraction (test) {l['majority_class_fraction_test']:.2f}")
    print(f"wrote {out / 'profile.json'}, {out / 'profile.png'}")


if __name__ == "__main__":
    main()
