"""sample.py — draw synthetic samples from a trained run.

  # unconditional model: N samples
  python tools/sample.py --run outputs/r3/flow --n 2000

  # conditional model: same number for every class
  python tools/sample.py --run outputs/r2/ddpm --n-per-class 500 --steps 100 --guidance 2.0

  # conditional model: chosen counts for chosen classes (e.g. class balancing)
  python tools/sample.py --run outputs/r1/cvae --class-counts '{"Megakaryocytes": 300, "Dendritic cells": 250}'

Sampler knobs (passed to the model, ignored where irrelevant):
  --temperature (vae/flow latent scale), --add-noise (vae: also draw observation noise),
  --steps / --guidance / --eta (ddpm DDIM sampler).
  --latent-source prior|posterior|interpolate (vae only): draw z from p(z) (default),
      from q(z|x) of random REAL training samples of the requested class, or by
      interpolating latents of two same-class training samples.

Writes into --run:  samples[_TAG].npz  (tabular: X = model-space rows; images:
X_uint8 = 8-bit pixels, rows in H,W,C order; y = class index), sample_config[_TAG].json, sample_grid[_TAG].png (images:
one row of synthetic + one row of real per class).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported (Apple GPU)

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Dataset, Prep, get_device, read_json, set_seed, to_flat, to_model_shape,  # noqa: E402
                    to_tensor, write_json)
from models import build_model  # noqa: E402


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["family"], **ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, Prep.load(run_dir / "prep.pkl"), read_json(run_dir / "run_config.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--data", default=None, help="defaults to data/<dataset from run_config>")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--n", type=int)
    g.add_argument("--n-per-class", type=int)
    g.add_argument("--class-counts", type=str)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--add-noise", action="store_true")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--latent-source", default="prior", choices=["prior", "posterior", "interpolate"])
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=500)
    a = p.parse_args()

    set_seed(a.seed)
    dev = get_device(a.device)
    model, prep, cfg = load_run(a.run, dev)
    root = Path(__file__).resolve().parent.parent
    ds = Dataset(a.data or root / "data" / cfg["dataset"])

    if a.n is not None:
        if model.conditional:
            sys.exit("conditional model: use --n-per-class or --class-counts")
        ys = np.zeros(a.n, dtype=np.int64)
        labelled = False
    else:
        if not model.conditional:
            sys.exit("unconditional model: use --n")
        if a.n_per_class:
            counts = {k: a.n_per_class for k in range(ds.n_classes)}
        else:
            raw = json.loads(a.class_counts)
            counts = {ds.resolve_classes(str(k))[0]: int(v) for k, v in raw.items()}
        ys = np.concatenate([np.full(v, k, dtype=np.int64) for k, v in counts.items()])
        labelled = True

    kw = dict(temperature=a.temperature, add_noise=a.add_noise, steps=a.steps, guidance=a.guidance, eta=a.eta)
    ref_idx = None
    if a.latent_source != "prior":
        if model.family != "vae":
            sys.exit("--latent-source posterior/interpolate is only defined for vae")
        inc = ds.resolve_classes(",".join(cfg.get("include_classes", [])))
        exc = ds.resolve_classes(",".join(cfg.get("exclude_classes", [])))
        pool = ds.indices("train", include=inc, exclude=exc)
        rng = np.random.default_rng(a.seed)
        if labelled:
            ref_idx = np.concatenate([rng.choice(pool[ds.y[pool] == k], (ys == k).sum(), replace=True)
                                      for k in np.unique(ys)])
            ys = ds.y[ref_idx].astype(np.int64)
        else:
            ref_idx = rng.choice(pool, len(ys), replace=True)
        Xref = to_model_shape(prep.transform(ds.X[ref_idx]), ds.image_shape, cfg["backbone"])
        kw = dict(temperature=a.temperature, add_noise=a.add_noise, mode=a.latent_source)
    t0, chunks = time.time(), []
    for b in range(0, len(ys), a.batch):
        yb = torch.from_numpy(ys[b:b + a.batch]).to(dev)
        if ref_idx is not None:
            xb = model.sample_from_reference(to_tensor(Xref[b:b + a.batch], dev),
                                             yb if model.conditional else None, **kw)
        else:
            xb = model.sample(len(yb), yb if model.conditional else None, **kw)
        chunks.append(xb.float().cpu().numpy())
    X = to_flat(np.concatenate(chunks), ds.image_shape)
    n_bad = int((~np.isfinite(X)).any(1).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    secs = round(time.time() - t0, 1)

    sfx = f"_{a.tag}" if a.tag else ""
    run = Path(a.run)
    if ds.modality == "image":   # images: store 8-bit pixels only (what evaluation and viewers use)
        save = {"X_uint8": prep.inverse_image(X), "y": ys, "labelled": labelled}
    else:
        save = {"X": X.astype(np.float32), "y": ys, "labelled": labelled}
    np.savez_compressed(run / f"samples{sfx}.npz", **save)
    write_json(run / f"sample_config{sfx}.json", {
        "n_samples": int(len(ys)), "labelled": labelled, "sampler_kwargs": kw,
        "latent_source": a.latent_source,
        "n_distinct_reference_samples": int(len(np.unique(ref_idx))) if ref_idx is not None else None, "seconds": secs,
        "non_finite_samples_replaced": n_bad, "device": str(dev),
        "class_counts": {ds.class_name(k): int((ys == k).sum()) for k in np.unique(ys)} if labelled else {},
    })

    if ds.modality == "image":
        H, W, C = ds.image_shape
        classes = list(np.unique(ys)) if labelled else [None]
        k = 10
        fig, axes = plt.subplots(len(classes), 2, figsize=(2 * k * 0.55, len(classes) * 0.75 + 0.4),
                                 squeeze=False)
        real_idx = ds.indices("test")
        for r, c in enumerate(classes):
            syn = save["X_uint8"][ys == c][:k] if c is not None else save["X_uint8"][:k]
            ri = real_idx[ds.y[real_idx] == c][:k] if c is not None else real_idx[:k]
            for col, block in enumerate([syn, ds.X[ri]]):
                tiles = [b.reshape(H, W, C) for b in block]
                tiles += [np.zeros((H, W, C), np.uint8)] * (k - len(tiles))
                axes[r, col].imshow(np.concatenate(tiles, 1).squeeze())
                axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
                if r == 0:
                    axes[r, col].set_title(["synthetic", "real (test)"][col], fontsize=9)
            if c is not None:
                axes[r, 0].set_ylabel(ds.class_name(c)[:18], rotation=0, ha="right", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(run / f"sample_grid{sfx}.png", dpi=120)
    print(f"[sample] {len(ys)} samples in {secs}s -> {run / f'samples{sfx}.npz'} (non-finite: {n_bad})")


if __name__ == "__main__":
    main()
