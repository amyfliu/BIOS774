"""train.py — fit one generative model on the TRAIN split of a dataset.

Usage (all decisions are passed in by the agent; this script makes none):

  python tools/train.py --data data/pbmc3k --out outputs/r1_pbmc3k/cvae_l16 \
      --model vae --backbone mlp --prep libnorm,log1p,hvg:2000,standardize \
      --conditional --epochs 200 --batch-size 128 --lr 1e-3 \
      --params '{"latent_dim": 16, "beta": 1.0}' --max-minutes 10

Models   : vae | wgan | flow | ddpm        (see tools/models/)
Backbone : mlp (tabular or flattened images) | conv (images only)
Prep     : see tools/common.py::Prep (fit on train only, saved as prep.pkl)
Subsets  : --include-classes / --exclude-classes (e.g. train only on "normal"
           classes for anomaly detection), --max-train N (stratified subsample)
Budget   : --max-minutes stops training cleanly at the time limit
Smoke    : --smoke = tiny data, 2 epochs, tiny network; checks the code path only

Writes to --out:
  model.pt, prep.pkl, run_config.json (every setting + data summary),
  train_log.json (per-epoch losses, final diagnostics, stop reason), loss.png
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported (Apple GPU)

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, Prep, Timer, get_device, set_seed, to_model_shape, to_tensor, write_json  # noqa: E402
from models import build_model  # noqa: E402

PLAIN_LABEL = {"loss_per_dim": "total loss (per feature)", "kl": "KL term (latent use)",
               "recon_mse": "reconstruction error", "critic_loss": "critic loss",
               "wasserstein_est": "Wasserstein estimate", "gp": "gradient penalty",
               "gen_loss": "generator loss", "nll_per_dim": "neg. log-likelihood (per dim)",
               "bits_per_dim": "bits per dimension", "loss": "denoising loss"}
FAMILY_NAME = {"vae": "VAE", "wgan": "WGAN-GP", "flow": "normalizing flow", "ddpm": "diffusion model (DDPM)"}

SMOKE_PARAMS = {"vae": {"latent_dim": 4, "hidden": [32], "ch": 8},
                "wgan": {"latent_dim": 8, "hidden": [32], "ch": 8, "n_critic": 2},
                "flow": {"n_layers": 2, "hidden": 16},
                "ddpm": {"T": 50, "ch": 8, "hidden": 32, "n_layers": 1}}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True, choices=["vae", "wgan", "flow", "ddpm"])
    p.add_argument("--backbone", default="mlp", choices=["mlp", "conv"])
    p.add_argument("--prep", default="none")
    p.add_argument("--conditional", action="store_true", help="condition on class labels")
    p.add_argument("--include-classes", default=None)
    p.add_argument("--exclude-classes", default=None)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--params", default="{}", help="JSON dict of model hyperparameters")
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = get_device(a.device)
    ds = Dataset(a.data)

    if a.backbone == "conv" and ds.modality != "image":
        sys.exit("conv backbone requires image data (prepared.npz with image_shape)")
    if a.model == "flow" and a.backbone == "conv" and "scale01" not in a.prep:
        sys.exit("conv flow models 8-bit pixels: use --prep scale01")

    inc, exc = ds.resolve_classes(a.include_classes), ds.resolve_classes(a.exclude_classes)
    max_train = 256 if a.smoke else a.max_train
    idx = ds.indices("train", include=inc, exclude=exc, max_n=max_train, seed=a.seed)
    prep = Prep(a.prep)
    Xtr = prep.fit_transform(ds.X[idx])
    Xtr = to_model_shape(Xtr, ds.image_shape, a.backbone)
    ytr = ds.y[idx] if ds.y is not None else np.zeros(len(idx), dtype=int)
    if a.conditional and ds.y is None:
        sys.exit("--conditional needs labels")

    params = json.loads(a.params)
    epochs = a.epochs
    if a.smoke:
        params = {**params, **SMOKE_PARAMS[a.model]}
        epochs = 2
    out_act = {"scale_pm1": "tanh", "scale01": "sigmoid"}.get(prep.steps[-1]) if ds.modality == "image" else None
    if a.model == "flow":
        out_act = None
    model = build_model(a.model, data_shape=Xtr.shape[1:], n_classes=ds.n_classes,
                        conditional=a.conditional, backbone=a.backbone, out_act=out_act, **params).to(dev)
    model.configure_optimizers(a.lr)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    X_t, y_t = to_tensor(Xtr, dev), torch.from_numpy(ytr.astype(np.int64)).to(dev)
    n = len(X_t)
    bs = min(a.batch_size, n)
    timer = Timer(a.max_minutes)
    log, stop_reason = [], "completed"
    print(f"[train] {a.model}/{a.backbone} on {dev} | n_train={n} shape={tuple(Xtr.shape[1:])} "
          f"params={n_params:,} | prep={a.prep}", flush=True)

    for ep in range(epochs):
        model.train()
        model.set_epoch(ep)
        perm = torch.randperm(n, device=dev)
        agg = {}
        for b in range(0, n - bs + 1, bs):
            bi = perm[b:b + bs]
            stats = model.train_step(X_t[bi], y_t[bi])
            for k, v in stats.items():
                agg.setdefault(k, []).append(v)
            if timer.out_of_time():
                break
        row = {"epoch": ep + 1, "seconds": round(timer.elapsed, 1),
               **{k: float(np.mean(v)) for k, v in agg.items()}}
        log.append(row)
        main_key = next(iter(agg))
        print(f"[train] epoch {ep + 1}/{epochs} {main_key}={row[main_key]:.4f} ({row['seconds']}s)", flush=True)
        if not all(math.isfinite(v) for k, v in row.items() if isinstance(v, float)):
            stop_reason = "diverged (non-finite loss)"
            break
        if timer.out_of_time():
            stop_reason = f"time budget reached ({a.max_minutes} min)"
            break

    model.eval()
    with torch.no_grad():
        sel = torch.randperm(n, device=dev)[:min(512, n)]
        try:
            diag = model.diagnostics(X_t[sel], y_t[sel])
        except Exception as e:  # diagnostics must never kill a finished run
            diag = {"error": repr(e)}

    torch.save({"family": a.model, "config": model.config(), "state_dict": model.state_dict()}, out / "model.pt")
    prep.save(out / "prep.pkl")
    write_json(out / "run_config.json", {
        "dataset": ds.name, "modality": ds.modality, "model": a.model, "backbone": a.backbone,
        "prep": prep.info(), "conditional": a.conditional,
        "include_classes": [ds.class_name(k) for k in inc], "exclude_classes": [ds.class_name(k) for k in exc],
        "n_train": int(n), "model_input_shape": list(Xtr.shape[1:]), "epochs_requested": epochs,
        "epochs_run": len(log), "batch_size": bs, "lr": a.lr, "params": model.params, "n_parameters": n_params,
        "max_minutes": a.max_minutes, "seed": a.seed, "device": str(dev), "smoke": a.smoke,
        "train_class_counts": {ds.class_name(k): int((ytr == k).sum()) for k in np.unique(ytr)} if ds.y is not None else {},
        "classes": ds.class_names(),
    })
    write_json(out / "train_log.json", {"stop_reason": stop_reason, "train_seconds": round(timer.elapsed, 1),
                                        "epochs": log, "final_diagnostics": diag})
    keys = [k for k in log[0] if k not in ("epoch", "seconds", "beta")] if log else []
    n_pan = max(1, len(keys))
    fig, axes = plt.subplots(1, n_pan, figsize=(max(6.0, 3.4 * n_pan), 3.2))
    for ax, k in zip(np.atleast_1d(axes), keys):
        ax.plot([r["epoch"] for r in log], [r.get(k, np.nan) for r in log])
        ax.set_title(PLAIN_LABEL.get(k, k), fontsize=9)
        ax.set_xlabel("epoch")
    fig.suptitle(f"Training curves: {FAMILY_NAME.get(a.model, a.model)}\n({a.backbone} network, {ds.name})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=110)
    print(f"[train] done: {stop_reason}; {round(timer.elapsed, 1)}s; diagnostics={json.dumps(diag)[:300]}")


if __name__ == "__main__":
    main()
