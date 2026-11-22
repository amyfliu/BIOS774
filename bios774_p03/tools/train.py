"""train.py — learn a representation from preprocessed modality features.

  python tools/train.py --data D --out O/<label> --inputs rna:log1p_hvg2000_pca50,atac:tfidf_lsi50_drop1 \
      --method clip --dim 32 --encoder mlp --epochs 200 --batch-size 512 --lr 1e-3 --max-minutes 5 [--smoke]

--inputs names the feature files written by preprocess.py, as modality:spec
(read from <features-dir>/features/, default = the parent of --out).

Methods (all fit on the train split; val drives early stopping):
  pca_concat  early fusion baseline, no learning: z-score each modality, weight
              modalities equally, concatenate, PCA to --dim. With ONE input it is
              the single-modality baseline. (lecture: PCA)
  identity    use the input features as the embedding, unchanged (e.g. a frozen
              pretrained encoder such as BiomedCLIP = zero-shot baseline)
  clip        one encoder per modality, trained so paired samples are nearest
              neighbours: symmetric InfoNCE with learnable temperature (L23/L24)
  vicreg      one encoder per modality; invariance (paired embeddings close) +
              variance + covariance terms; no negatives needed (L24)
  supcon_clip clip loss + supervised contrastive loss on --label: samples of the
              same class are also positives (L23). Uses labels for training.
  simclr      ONE modality, two random views (feature masking + noise), InfoNCE
              between views; self-supervised single-modality encoder (L24)

Encoders (per modality): linear | mlp (--hidden, --layers, --dropout) |
adapter (residual MLP added to the input, last layer initialised at zero so
training STARTS at the input features; needs input dim == --dim; the light
fine-tuning option for pretrained features).

Writes O/<label>/: embeddings.npz (z_joint + z_<modality> for every sample),
model.pt, run_config.json, train_log.json, loss.png, TRAINING_DONE.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, feature_path, l2n, load_features, rounded, save_json, set_seed  # noqa: E402

METHODS = ["pca_concat", "identity", "clip", "vicreg", "supcon_clip", "simclr"]
LEARNED = {"clip", "vicreg", "supcon_clip", "simclr"}


# ----------------------------------------------------------------------------- inputs
def load_inputs(spec: str, fdir: Path):
    mods = {}
    for item in spec.split(","):
        mod, s = item.split(":", 1)
        X, info = load_features(feature_path(fdir, mod.strip(), s.strip()))
        mods[mod.strip()] = (X, info)
    return mods


def standardise(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0)
    return ((X - mu) / np.where(sd > 1e-8, sd, 1)).astype(np.float32)


# ----------------------------------------------------------------------------- no-training methods
def run_pca_concat(mods, tr, dim):
    from sklearn.decomposition import PCA

    blocks = [standardise(X, tr) / math.sqrt(X.shape[1]) for X, _ in mods.values()]
    A = np.hstack(blocks)
    k = min(dim, A.shape[1], len(tr) - 1)
    pca = PCA(n_components=k, random_state=0).fit(A[tr])
    Z = pca.transform(A).astype(np.float32)
    emb = {"z_joint": Z}
    if len(mods) == 1:
        emb[f"z_{next(iter(mods))}"] = Z
    return emb, {"explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()), "dim": k}


def run_identity(mods):
    emb = {f"z_{m}": X.astype(np.float32) for m, (X, _) in mods.items()}
    dims = {X.shape[1] for X, _ in mods.values()}
    if len(dims) == 1:
        emb["z_joint"] = l2n(np.mean([l2n(X) for X, _ in mods.values()], axis=0)).astype(np.float32)
    else:
        emb["z_joint"] = np.hstack([l2n(X) for X, _ in mods.values()]).astype(np.float32)
    return emb, {}


# ----------------------------------------------------------------------------- torch parts
def build_encoder(kind, d_in, d_out, hidden, layers, dropout):
    import torch.nn as nn

    if kind == "linear":
        return nn.Linear(d_in, d_out)
    if kind == "mlp":
        mods, d = [], d_in
        for _ in range(max(layers, 1)):
            mods += [nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, d_out))
        return nn.Sequential(*mods)
    if kind == "adapter":
        if d_in != d_out:
            raise SystemExit(f"adapter needs input dim == --dim (got {d_in} vs {d_out})")

        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(hidden, d_in))
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def forward(self, x):
                return x + self.net(x)

        return Adapter()
    raise SystemExit(f"unknown encoder {kind}")


def info_nce(za, zb, logit_scale):
    import torch
    import torch.nn.functional as F

    za, zb = F.normalize(za, dim=1), F.normalize(zb, dim=1)
    logits = logit_scale.exp().clamp(max=100) * za @ zb.t()
    tgt = torch.arange(len(za), device=za.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def supcon(z, y, temperature=0.1):
    """Supervised contrastive loss (Khosla et al.) over a batch of embeddings."""
    import torch
    import torch.nn.functional as F

    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    n = len(z)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    npos = pos.sum(1)
    ok = npos > 0
    loss = -(logp * pos).sum(1)[ok] / npos[ok]
    return loss.mean()


def vicreg_loss(za, zb, lam=25.0, mu=25.0, nu=1.0):
    import torch
    import torch.nn.functional as F

    inv = F.mse_loss(za, zb)

    def var_cov(z):
        z = z - z.mean(0)
        std = torch.sqrt(z.var(0) + 1e-4)
        v = F.relu(1 - std).mean()
        c = (z.t() @ z) / max(len(z) - 1, 1)
        off = c - torch.diag(torch.diag(c))
        return v, (off ** 2).sum() / z.shape[1]

    va, ca = var_cov(za)
    vb, cb = var_cov(zb)
    parts = {"invariance": inv.item(), "variance": (va + vb).item(), "covariance": (ca + cb).item()}
    return lam * inv + mu * (va + vb) + nu * (ca + cb), parts


def augment(x, mask_p, noise):
    import torch

    keep = (torch.rand_like(x) > mask_p).float()
    return x * keep + noise * torch.randn_like(x)


def recall_at_1(za, zb):
    za, zb = l2n(za), l2n(zb)
    s = za @ zb.T
    return float((s.argmax(1) == np.arange(len(za))).mean())


def run_learned(a, mods, ds, tr, va, device, out):
    import torch

    torch.manual_seed(a.seed)
    names = list(mods)
    if a.method == "simclr" and len(names) != 1:
        raise SystemExit("simclr takes exactly one input modality")
    if a.method != "simclr" and len(names) != 2:
        raise SystemExit(f"{a.method} needs exactly two input modalities")
    X = {m: torch.from_numpy(np.ascontiguousarray(mods[m][0], dtype=np.float32)) for m in names}
    enc = torch.nn.ModuleDict({m: build_encoder(a.encoder, X[m].shape[1], a.dim, a.hidden, a.layers, a.dropout)
                               for m in names}).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / a.temperature), device=device))
    params = list(enc.parameters()) + ([logit_scale] if a.learn_temperature else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    y = None
    if a.method == "supcon_clip":
        codes, classes = ds.labels(a.label)
        if codes is None:
            raise SystemExit(f"supcon_clip needs a label column (--label); have {ds.label_columns}")
        y = torch.from_numpy(codes).long()

    def loss_on(idx, train=True):
        xb = {m: X[m][idx].to(device) for m in names}
        parts = {}
        if a.method == "simclr":
            m = names[0]
            z1 = enc[m](augment(xb[m], a.mask_p, a.noise))
            z2 = enc[m](augment(xb[m], a.mask_p, a.noise))
            loss = info_nce(z1, z2, logit_scale)
        else:
            za, zb = enc[names[0]](xb[names[0]]), enc[names[1]](xb[names[1]])
            if a.method == "vicreg":
                loss, parts = vicreg_loss(za, zb)
            else:
                loss = info_nce(za, zb, logit_scale)
                parts["clip"] = loss.item()
                if a.method == "supcon_clip":
                    yb = y[idx].to(device)
                    sc = supcon(torch.cat([za, zb]), torch.cat([yb, yb]), a.supcon_temperature)
                    parts["supcon"] = sc.item()
                    loss = loss + a.supcon_weight * sc
        return loss, parts

    def embed(model_eval=True):
        enc.eval()
        out_ = {}
        with torch.no_grad():
            for m in names:
                zs = [enc[m](X[m][s:s + 4096].to(device)).cpu().numpy() for s in range(0, len(X[m]), 4096)]
                out_[m] = np.concatenate(zs).astype(np.float32)
        enc.train()
        return out_

    rng = np.random.default_rng(a.seed)
    log, best, best_state, bad = [], float("inf"), None, 0
    t0 = time.time()
    epochs = 2 if a.smoke else a.epochs
    tr_use = tr[: min(len(tr), 256)] if a.smoke else tr
    bs = min(a.batch_size, len(tr_use))
    stop_reason = "epochs"
    for ep in range(1, epochs + 1):
        perm = rng.permutation(tr_use)
        tot, n, agg = 0.0, 0, {}
        for s in range(0, len(perm) - bs + 1, bs):  # drop last partial batch (contrastive needs full batches)
            idx = perm[s:s + bs]
            loss, parts = loss_on(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += loss.item() * len(idx); n += len(idx)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * len(idx)
        enc.eval()
        with torch.no_grad():
            vl = []
            for s in range(0, max(len(va) - bs + 1, 1), bs):
                idx = va[s:s + bs]
                if len(idx) >= 2:
                    vl.append(loss_on(idx)[0].item())
        enc.train()
        val_loss = float(np.mean(vl)) if vl else float("nan")
        rec = {"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": val_loss,
               "temperature": float(1 / logit_scale.exp().item()), "seconds": round(time.time() - t0, 1)}
        rec.update({f"train_{k}": v / max(n, 1) for k, v in agg.items()})
        if ep == 1 or ep % a.eval_every == 0 or ep == epochs:
            Z = embed()
            zs = [Z[m][va] for m in names]
            rec["val_embedding_sd_mean"] = float(np.mean([z.std(0).mean() for z in zs]))
            if len(names) == 2:
                sub = va[: min(len(va), 1000)]
                rec["val_recall_at_1"] = recall_at_1(Z[names[0]][sub], Z[names[1]][sub])
                rec["val_recall_at_1_chance"] = 1 / len(sub)
        log.append(rec)
        if ep == 1 or ep % max(1, a.eval_every) == 0:
            extra = f" val R@1 {rec['val_recall_at_1']:.3f}" if "val_recall_at_1" in rec else ""
            print(f"  epoch {ep:4d}  train {rec['train_loss']:.4f}  val {val_loss:.4f}{extra}  ({rec['seconds']:.0f}s)")
        if np.isfinite(val_loss) and val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
            best_epoch = ep
        else:
            bad += 1
        if not np.isfinite(rec["train_loss"]):
            stop_reason = "non-finite loss"; break
        if bad >= a.patience:
            stop_reason = f"early stop (val loss not improving for {a.patience} epochs)"; break
        if (time.time() - t0) / 60 > a.max_minutes:
            stop_reason = "time budget"; break
    if best_state is not None:
        enc.load_state_dict(best_state)
    else:
        best_epoch = len(log)
    Z = embed()
    if len(names) == 2:  # matching quality of the KEPT (best-val) model
        sub = va[: min(len(va), 1000)]
        best_val_r1 = recall_at_1(Z[names[0]][sub], Z[names[1]][sub])
    else:
        best_val_r1 = None
    emb = {f"z_{m}": Z[m] for m in names}
    emb["z_joint"] = Z[names[0]] if len(names) == 1 else l2n(np.mean([l2n(Z[m]) for m in names], 0)).astype(np.float32)
    torch.save({"encoders": enc.state_dict(), "logit_scale": logit_scale.detach().cpu()}, out / "model.pt")
    summary = {"epochs_run": len(log), "best_epoch": best_epoch, "best_val_loss": best,
               "stop_reason": stop_reason, "val_recall_at_1_kept_model": best_val_r1,
               "minutes": round((time.time() - t0) / 60, 2),
               "n_parameters": int(sum(p.numel() for p in enc.parameters()))}
    return emb, log, summary


def plot_log(log, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in log]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(ep, [r["train_loss"] for r in log], label="train")
    axes[0].plot(ep, [r["val_loss"] for r in log], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_title("loss"); axes[0].legend()
    pts = [r for r in log if "val_recall_at_1" in r]
    if pts:
        axes[1].plot([r["epoch"] for r in pts], [r["val_recall_at_1"] for r in pts], "o-", label="val recall@1")
        axes[1].axhline(pts[0]["val_recall_at_1_chance"], ls="--", c="grey", label="chance")
        axes[1].set_title("cross-modal matching (val)")
    pts = [r for r in log if "val_embedding_sd_mean" in r]
    ax2 = axes[1].twinx()
    ax2.plot([r["epoch"] for r in pts], [r["val_embedding_sd_mean"] for r in pts], "s:", c="C3",
             label="embedding sd (collapse check)")
    ax2.set_ylabel("mean sd", color="C3")
    axes[1].set_xlabel("epoch")
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="run folder for THIS model, e.g. outputs/<run_id>/clip_mlp_d32")
    ap.add_argument("--inputs", required=True, help="modality:spec[,modality:spec]")
    ap.add_argument("--features-dir", default=None, help="folder holding features/ (default: parent of --out)")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--encoder", default="mlp", choices=["linear", "mlp", "adapter"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07, help="initial InfoNCE temperature")
    ap.add_argument("--fixed-temperature", dest="learn_temperature", action="store_false")
    ap.add_argument("--label", default=None, help="label column for supcon_clip (default: primary)")
    ap.add_argument("--supcon-weight", type=float, default=0.5)
    ap.add_argument("--supcon-temperature", type=float, default=0.1)
    ap.add_argument("--mask-p", type=float, default=0.3, help="simclr: fraction of features masked per view")
    ap.add_argument("--noise", type=float, default=0.1, help="simclr: gaussian noise sd per view")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 256 samples, to catch errors")
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "TRAINING_DONE").unlink(missing_ok=True)
    fdir = Path(a.features_dir) if a.features_dir else out.parent
    ds = Dataset(a.data)
    mods = load_inputs(a.inputs, fdir)
    tr, va = ds.idx("train"), ds.idx("val")
    t0 = time.time()
    log, summary = [], {}
    if a.method == "pca_concat":
        emb, summary = run_pca_concat(mods, tr, a.dim)
    elif a.method == "identity":
        emb, summary = run_identity(mods)
    else:
        from common import get_device

        device = get_device(a.device)
        print(f"{a.method} on {device}: inputs {[(m, X.shape[1]) for m, (X, _) in mods.items()]}, dim {a.dim}")
        emb, log, summary = run_learned(a, mods, ds, tr, va, device, out)
        save_json(rounded(log), out / "train_log.json")
        plot_log(log, out / "loss.png")
    np.savez_compressed(out / "embeddings.npz", **emb)
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update(inputs_resolved={m: {"spec": i.get("spec"), "dim": int(X.shape[1]), "steps": i.get("steps", [])}
                                for m, (X, i) in mods.items()},
               embedding_keys=list(emb), embedding_dim={k: int(v.shape[1]) for k, v in emb.items()},
               uses_labels=a.method == "supcon_clip", summary=summary,
               seconds=round(time.time() - t0, 1))
    if a.method not in LEARNED:
        for k in ("encoder", "hidden", "layers", "dropout", "epochs", "batch_size", "lr", "weight_decay",
                  "temperature", "learn_temperature", "supcon_weight", "supcon_temperature", "mask_p", "noise",
                  "patience", "eval_every", "max_minutes"):
            cfg.pop(k, None)
    save_json(rounded(cfg), out / "run_config.json")
    (out / "TRAINING_DONE").write_text("ok\n")
    print(f"done: {a.method} -> {out}  embeddings {cfg['embedding_dim']}  {summary}")


if __name__ == "__main__":
    main()
