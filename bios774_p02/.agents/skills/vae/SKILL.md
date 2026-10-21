---
name: vae
description: Configure, train, check and sample a (conditional) VAE with tools/train.py --model vae.
---

# Model (tools/models/vae.py)
Encoder q(z|x[,y]) → Gaussian latent; decoder p(x|z[,y]) Gaussian with a
learned per-feature variance, so the ELBO is a real lower bound on log p(x)
(usable by `score.py`). Loss = −ELBO with a β weight on the KL term and a
linear KL warm-up.

# Hyperparameters (`--params` JSON) — starting points
| param | tabular (n ≈ 10³) | images 28×28 |
|---|---|---|
| `latent_dim` | 8–16 (≈ number of biological axes; more for more classes) | 16–32 |
| `hidden` (mlp) | [512, 256] | — |
| `ch` (conv) | — | 32 |
| `beta` | 1.0 (lower, e.g. 0.5, if samples are too smooth) | 1.0 |
| `kl_warmup_epochs` | 10–20 | 3–5 |
Other: `--lr 1e-3`, `--batch-size 64–128`; epochs: tabular 100–300 (fast);
images 10–30 on a subsample.

# Health checks (train_log.json → final_diagnostics)
- `posterior_collapse_suspected` / `active_units`: if 0 active units or KL
  ≈ 0, the decoder ignores z → lower `beta`, lengthen warm-up, or shrink
  the decoder.
- `kl_per_latent_dim`: a few dims near 0 is normal (unused capacity).
- Loss curve flattening = converged; still falling at the end = could
  train longer if budget allows.

# Sampling (tools/sample.py)
- Default: z ~ p(z) = N(0, I), decoded with the decoder MEAN. Means are
  smoother than real data: expect high precision, low recall. For tabular
  data `--add-noise` also draws the observation noise (wider spread; check
  that precision does not collapse).
- **Rare classes: check label consistency.** If a conditional VAE's
  prior samples for a class are classified as that class much less often
  than real data is (`metrics → fidelity.label_consistency.per_class`), the
  class's latent region does not line up with the prior — typical when it
  has only tens of training samples. Re-sample with
  `--latent-source posterior` (z ~ q(z|x) of random real training samples
  of that class) or `--latent-source interpolate` (random interpolations of
  two same-class training latents), and log it as an `iteration`.
  These are anchored to real training samples by construction: they will
  sit closer to training data (memorization metrics will show it). That is
  acceptable for balancing/augmentation; it is NOT acceptable for a
  privacy-motivated release.
- `--temperature` < 1 trades diversity for typicality.

# Typical outputs to report
loss curve, active units / KL, label consistency per class, precision vs
recall, and which latent source was used.
