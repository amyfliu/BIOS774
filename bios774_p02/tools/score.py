"""score.py — per-sample log-likelihood of REAL data under a trained model.

Used for density-based tasks (anomaly / out-of-distribution detection,
"how typical is this sample?"). Only models with a likelihood support it:
  flow -> exact log p(x);   vae -> ELBO (a lower bound on log p(x)).
GANs and DDPMs (as implemented here) have no per-sample likelihood.

For a conditional model the label is NOT assumed known: the score is the
marginal log p(x) = logsumexp_k [log p(x|k) + log pi_k], with pi = the class
frequencies the model was trained on.

  python tools/score.py --run outputs/r3/flow --split test

Writes scores_<split>.npz into --run: logp (per sample), logp_per_dim,
y (class index), idx (row index into prepared.npz).
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch is imported (Apple GPU)

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Dataset, get_device, to_model_shape, to_tensor  # noqa: E402
from sample import load_run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--data", default=None)
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=500)
    a = p.parse_args()

    dev = get_device(a.device)
    model, prep, cfg = load_run(a.run, dev)
    if not model.supports_log_prob:
        sys.exit(f"{model.family} has no per-sample likelihood; use a flow or vae for density scoring")
    root = Path(__file__).resolve().parent.parent
    ds = Dataset(a.data or root / "data" / cfg["dataset"])
    idx = ds.indices(a.split, max_n=a.max_n)
    X = to_model_shape(prep.transform(ds.X[idx]), ds.image_shape, cfg["backbone"])
    D = int(np.prod(X.shape[1:]))

    if model.conditional:
        names = cfg["classes"]
        tc = cfg["train_class_counts"]
        prior = np.array([tc.get(n, 0) for n in names], dtype=np.float64)
        log_pi = np.log(np.maximum(prior / prior.sum(), 1e-12))
    out = []
    torch.manual_seed(0)
    for b in range(0, len(X), a.batch):
        xb = to_tensor(X[b:b + a.batch], dev)
        if not model.conditional:
            out.append(model.log_prob(xb).cpu().numpy())
        else:
            per = []
            for k in range(model.n_classes):
                if not np.isfinite(log_pi[k]) or log_pi[k] < -27:
                    continue
                yk = torch.full((len(xb),), k, device=dev, dtype=torch.long)
                per.append(model.log_prob(xb, yk).cpu().numpy() + log_pi[k])
            per = np.stack(per, 1)
            m = per.max(1, keepdims=True)
            out.append((m + np.log(np.exp(per - m).sum(1, keepdims=True))).squeeze(1))
    logp = np.concatenate(out).astype(np.float64)
    np.savez_compressed(Path(a.run) / f"scores_{a.split}.npz", logp=logp, logp_per_dim=logp / D,
                        y=ds.y[idx] if ds.y is not None else np.zeros(len(idx)), idx=idx)
    print(f"[score] {len(idx)} {a.split} samples; mean log p per dim = {np.mean(logp) / D:.4f}")


if __name__ == "__main__":
    main()
