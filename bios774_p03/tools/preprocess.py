"""preprocess.py — turn one modality into a numeric feature matrix (fit on train only).

  python tools/preprocess.py --data D --out O --modality rna  --spec log1p_hvg2000_pca50
  python tools/preprocess.py --data D --out O --modality atac --spec tfidf_lsi50_drop1
  python tools/preprocess.py --data D --out O --modality image --spec biomedclip
  python tools/preprocess.py --data D --out O --modality text  --spec tfidf_svd128

Writes O/features/<modality>__<spec>.npz (X: [all samples, d], float32) and a
.json next to it (what was done, dims, fit statistics). Every learning method in
tools/train.py reads these matrices, so the spec is THE preprocessing decision.
Any fitted step (feature selection, IDF weights, PCA/SVD, scaling) is fitted on
the train split and then applied to val/test.

Spec = steps joined by "_", applied left to right.

counts modalities (raw counts):
  log1p        normalise each sample to 10,000 total counts, then log(1+x)
  tfidf        term-frequency x inverse-document-frequency, then log(1+1e4*x)
               (standard for sparse, near-binary data such as ATAC peaks)
  binary       x > 0 -> 1
  hvgN         keep the N most variable features (dispersion within mean bins)
  topN         keep the N features detected in the most train samples
  scale        z-score each feature (clipped at +-10)
  pcaK         PCA to K components
  lsiK         truncated SVD to K components, each component z-scored
  drop1        drop the first SVD/PCA component (in TF-IDF/LSI it mostly tracks
               sequencing depth, not biology)
  (no reduction step -> the full selected feature matrix, e.g. as a prediction target)

image modalities:
  biomedclip   frozen BiomedCLIP image encoder (ViT-B/16, PubMed figures) -> 512-d
  pixelsS      grayscale, resized to S x S, flattened (a deliberately weak baseline)
  ... optionally followed by _pcaK / _scale

text modalities:
  biomedclip   frozen BiomedCLIP text encoder (PubMedBERT, 256 tokens) -> 512-d
  tfidf_svdK   word TF-IDF (1-2 grams, words in >= 3 train texts) + truncated SVD
  ... optionally followed by _scale

BiomedCLIP image and text features come from ONE jointly pretrained model, so
they already live in a shared space (zero-shot alignment). Downloads the model
weights on first use (~800 MB) and caches the features in outputs/_cache/.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, feature_path, get_device, save_features  # noqa: E402

BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CACHE = Path(__file__).resolve().parent.parent / "outputs" / "_cache"
TOKEN = re.compile(r"^([a-z]+?)(\d*)$")


def parse(spec: str):
    steps = []
    for tok in spec.split("_"):
        if tok in ("log1p",):
            steps.append((tok, None))
            continue
        m = TOKEN.match(tok)
        if not m:
            raise SystemExit(f"bad spec step '{tok}' in '{spec}'")
        name, num = m.group(1), m.group(2)
        steps.append((name, int(num) if num else None))
    return steps


# ----------------------------------------------------------------------------- counts
def _hvg(X, tr, n, n_bins=20):
    """Dispersion (var/mean) z-scored within bins of mean expression, on train rows."""
    Xt = X[tr]
    mean = np.asarray(Xt.mean(0)).ravel() if hasattr(Xt, "toarray") else Xt.mean(0)
    sq = np.asarray(Xt.multiply(Xt).mean(0)).ravel() if hasattr(Xt, "multiply") else (Xt ** 2).mean(0)
    var = np.maximum(sq - mean ** 2, 0)
    disp = np.log(np.maximum(var, 1e-12) / np.maximum(mean, 1e-12))
    ok = mean > 0
    z = np.full(len(mean), -np.inf)
    if ok.sum():
        edges = np.unique(np.quantile(mean[ok], np.linspace(0, 1, n_bins + 1)))
        b = np.clip(np.searchsorted(edges, mean, side="right") - 1, 0, len(edges) - 2)
        for k in np.unique(b[ok]):
            sel = ok & (b == k)
            d = disp[sel]
            sd = d.std()
            z[sel] = (d - d.mean()) / sd if sd > 0 else 0.0
    keep = np.argsort(-z)[: min(n, int(ok.sum()))]
    return np.sort(keep)


def counts_pipeline(X, tr, steps, info):
    import scipy.sparse as sp
    from sklearn.decomposition import PCA, TruncatedSVD

    X = sp.csr_matrix(X, dtype=np.float32)
    dense = None  # becomes an ndarray once reduced
    last_reducer = None
    for name, num in steps:
        if name == "log1p":
            lib = np.asarray(X.sum(1)).ravel()
            X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
            X = X.log1p() if hasattr(X, "log1p") else sp.csr_matrix(np.log1p(X.toarray()))
            info["steps"].append("normalise to 1e4 per sample + log1p")
        elif name == "tfidf":
            lib = np.asarray(X.sum(1)).ravel()
            tf = sp.diags(1 / np.maximum(lib, 1)) @ X
            df = np.asarray((X[tr] > 0).sum(0)).ravel()
            idf = len(tr) / np.maximum(df, 1)
            X = (tf @ sp.diags(idf.astype(np.float32))).tocsr()
            X.data = np.log1p(X.data * 1e4)
            info["steps"].append("TF-IDF (IDF from train) + log1p(1e4 * x)")
        elif name == "binary":
            X = X.copy(); X.data = (X.data > 0).astype(np.float32)
            info["steps"].append("binarise (x>0)")
        elif name in ("hvg", "top"):
            if dense is not None:
                raise SystemExit(f"{name} must come before pca/lsi")
            if name == "hvg":
                keep = _hvg(X, tr, num)
            else:
                det = np.asarray((X[tr] > 0).sum(0)).ravel()
                keep = np.sort(np.argsort(-det)[:num])
            X = X[:, keep]
            info["selected_features"] = keep.tolist()
            info["steps"].append(f"keep {len(keep)} {'most variable' if name == 'hvg' else 'most detected'} features (train)")
        elif name == "scale":
            A = dense if dense is not None else X.toarray()
            mu, sd = A[tr].mean(0), A[tr].std(0)
            dense = np.clip((A - mu) / np.where(sd > 0, sd, 1), -10, 10)
            info["steps"].append("z-score per feature (train mean/sd), clip +-10")
        elif name in ("pca", "lsi"):
            A = dense if dense is not None else X
            k = min(num, A.shape[1] - 1, len(tr) - 1)
            if name == "pca":
                if sp.issparse(A):
                    A = A.toarray()
                red = PCA(n_components=k, random_state=0).fit(A[tr])
                dense = red.transform(A).astype(np.float32)
                info["explained_variance_ratio_sum"] = float(red.explained_variance_ratio_.sum())
            else:
                red = TruncatedSVD(n_components=k, random_state=0).fit(A[tr])
                Z = red.transform(A)
                dense = ((Z - Z[tr].mean(0)) / np.maximum(Z[tr].std(0), 1e-8)).astype(np.float32)
                info["explained_variance_ratio_sum"] = float(red.explained_variance_ratio_.sum())
            last_reducer = name
            info["steps"].append(f"{name.upper()} to {k} components (fit on train)")
        elif name == "drop":
            if dense is None or last_reducer is None:
                raise SystemExit("drop1 must follow pca/lsi")
            lib = np.asarray(X.sum(1)).ravel()
            r = np.corrcoef(dense[tr, 0], lib[tr])[0, 1]
            info["first_component_corr_with_depth"] = float(r)
            dense = dense[:, num or 1:]
            info["steps"].append(f"drop first {num or 1} component(s) (corr with depth r={r:.2f})")
        else:
            raise SystemExit(f"step '{name}' not valid for a counts modality")
    out = dense if dense is not None else X.toarray()
    return out.astype(np.float32)


# ----------------------------------------------------------------------------- pretrained (BiomedCLIP)
def load_biomedclip(device):
    try:
        import open_clip
    except ImportError:
        raise SystemExit("pip install open_clip_torch transformers   (needed for biomedclip)")
    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP)
    tokenizer = open_clip.get_tokenizer(BIOMEDCLIP)
    return model.to(device).eval(), preprocess, tokenizer


def biomedclip_features(ds: Dataset, mod: str, device, batch: int) -> np.ndarray:
    import torch
    from PIL import Image

    device = get_device(device)
    cache = CACHE / ds.name / f"{mod}__biomedclip.npz"
    key = np.array([f"{len(ds)}:{ds.ids[0]}:{ds.ids[-1]}"])
    if cache.exists():
        d = np.load(cache)
        if d["key"][0] == key[0]:
            print(f"  using cached {cache}")
            return d["X"]
    model, preprocess, tokenizer = load_biomedclip(device)
    out, t0 = [], time.time()
    n = len(ds)
    with torch.no_grad():
        if ds.kind(mod) == "image":
            paths = ds.image_paths(mod)
            for s in range(0, n, batch):
                x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[s:s + batch]]).to(device)
                out.append(model.encode_image(x).float().cpu().numpy())
                if (s // batch) % 20 == 0:
                    print(f"    {min(s + batch, n)}/{n} images  ({time.time() - t0:.0f}s)")
        else:
            texts = ds.texts(mod)
            for s in range(0, n, batch):
                tok = tokenizer(texts[s:s + batch], context_length=256).to(device)
                out.append(model.encode_text(tok).float().cpu().numpy())
                if (s // batch) % 20 == 0:
                    print(f"    {min(s + batch, n)}/{n} texts  ({time.time() - t0:.0f}s)")
    X = np.concatenate(out).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, key=key)
    return X


def dense_tail(A, tr, steps, info):
    from sklearn.decomposition import PCA

    for name, num in steps:
        if name == "pca":
            k = min(num, A.shape[1] - 1, len(tr) - 1)
            red = PCA(n_components=k, random_state=0).fit(A[tr])
            A = red.transform(A)
            info["explained_variance_ratio_sum"] = float(red.explained_variance_ratio_.sum())
            info["steps"].append(f"PCA to {k} (fit on train)")
        elif name == "scale":
            mu, sd = A[tr].mean(0), A[tr].std(0)
            A = (A - mu) / np.where(sd > 0, sd, 1)
            info["steps"].append("z-score per feature (train)")
        else:
            raise SystemExit(f"step '{name}' not valid here")
    return A.astype(np.float32)


# ----------------------------------------------------------------------------- image / text
def image_pipeline(ds, mod, tr, steps, info, device, batch):
    from PIL import Image

    head, num = steps[0]
    if head == "biomedclip":
        A = biomedclip_features(ds, mod, device, batch)
        info["steps"].append("frozen BiomedCLIP image encoder (512-d, shared space with its text encoder)")
        info["pretrained"] = BIOMEDCLIP
    elif head == "pixels":
        s = num or 32
        A = np.stack([np.asarray(Image.open(p).convert("L").resize((s, s)), dtype=np.float32).ravel() / 255
                      for p in ds.image_paths(mod)])
        info["steps"].append(f"grayscale {s}x{s} pixels, flattened")
    else:
        raise SystemExit(f"image spec must start with biomedclip or pixelsS, got '{head}'")
    return dense_tail(A, tr, steps[1:], info)


def text_pipeline(ds, mod, tr, steps, info, device, batch):
    head, num = steps[0]
    if head == "biomedclip":
        A = biomedclip_features(ds, mod, device, batch)
        info["steps"].append("frozen BiomedCLIP text encoder (512-d, shared space with its image encoder)")
        info["pretrained"] = BIOMEDCLIP
        rest = steps[1:]
    elif head == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if len(steps) < 2 or steps[1][0] != "svd":
            raise SystemExit("text tfidf must be followed by svdK, e.g. tfidf_svd128")
        texts = ds.texts(mod)
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_features=30000,
                              sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b")
        vec.fit([texts[i] for i in tr])
        M = vec.transform(texts)
        k = min(steps[1][1], M.shape[1] - 1, len(tr) - 1)
        svd = TruncatedSVD(n_components=k, random_state=0).fit(M[tr])
        A = svd.transform(M)
        info["vocabulary_size"] = int(len(vec.vocabulary_))
        info["explained_variance_ratio_sum"] = float(svd.explained_variance_ratio_.sum())
        info["steps"].append(f"word TF-IDF (1-2 grams, vocab {len(vec.vocabulary_)}, train) + SVD to {k}")
        rest = steps[2:]
    else:
        raise SystemExit(f"text spec must start with biomedclip or tfidf, got '{head}'")
    return dense_tail(A, tr, rest, info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="run output folder (features/ is created inside)")
    ap.add_argument("--modality", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64, help="batch size for pretrained encoders")
    a = ap.parse_args()

    ds = Dataset(a.data)
    if a.modality not in ds.modalities:
        raise SystemExit(f"unknown modality {a.modality}; have {list(ds.modalities)}")
    tr = ds.idx("train")
    steps = parse(a.spec)
    kind = ds.kind(a.modality)
    info = {"modality": a.modality, "kind": kind, "spec": a.spec, "steps": [], "fit_on": "train",
            "n_train": int(len(tr))}
    t0 = time.time()
    if kind == "counts":
        X = counts_pipeline(ds.counts(a.modality), tr, steps, info)
        if "selected_features" in info:
            fe = ds.features(a.modality)
            info["selected_feature_ids"] = fe.iloc[info["selected_features"], 0].astype(str).tolist()
            del info["selected_features"]
    elif kind == "image":
        X = image_pipeline(ds, a.modality, tr, steps, info, a.device, a.batch)
    else:
        X = text_pipeline(ds, a.modality, tr, steps, info, a.device, a.batch)
    if not np.isfinite(X).all():
        raise SystemExit("non-finite values after preprocessing")
    info.update(dim=int(X.shape[1]), seconds=round(time.time() - t0, 1),
                train_feature_sd_median=float(np.median(X[tr].std(0))))
    path = feature_path(a.out, a.modality, a.spec)
    save_features(path, X, info)
    print(f"{a.modality} [{a.spec}] -> {X.shape[0]} x {X.shape[1]}  ({info['seconds']}s)  {path}")
    for s in info["steps"]:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
